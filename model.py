import numpy as np
import random as r
from numba import jit, prange
from simulation import Simulation, record_time, template_params
import backend
# from pythonabm import Simulation, record_time, template_params
# from pythonabm.backend import record_time, check_direct, template_params, check_existing, get_end_step, Graph, \
#     progress_bar, starting_params, check_output_dir, assign_bins_jit, get_neighbors_cpu, get_neighbors_gpu

@jit(nopython=True, parallel=True)
def get_neighbor_forces(number_edges, edges, edge_forces, locations, center, direction_vectors, types, radius,
                        U_gg=40, U_cc=1, U_gc=30, U_gc_rep=70, r_e=1.01, u_repulsion=10000):
    interaction_strength = [U_gg, U_gc, U_cc]
    for index in range(number_edges):
        # get indices of cells in edge
        cell_1 = edges[index][0]
        cell_2 = edges[index][1]
        # get cell positions
        cell_1_loc = locations[cell_1] - center
        cell_2_loc = locations[cell_2] - center

        # get new location position
        vec = cell_2_loc - cell_1_loc
        dist2 = vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2

        # based on the distance apply force differently
        if dist2 == 0:
            edge_forces[index][0] = 0
            edge_forces[index][1] = 0
        else:
            dist = dist2 ** (1/2)
            if 0 < dist2 < (2 * radius) ** 2:
                edge_forces[index][0] = -1 * u_repulsion * (vec / dist)
                edge_forces[index][1] = 1 * u_repulsion * (vec / dist)
            else:
                # get the cell type
                cell_1_type = types[cell_1]
                cell_2_type = types[cell_2]
                if cell_1_type != cell_2_type:
                    # direction = [-1, 1]
                    # u = [20, 20]
                    # perform angle check:
                    if cell_1_type == 1:
                        direction_vec = direction_vectors[cell_1] - cell_1_loc
                        temp_vec = vec
                    else:
                        direction_vec = direction_vectors[cell_2] - cell_2_loc
                        temp_vec = -1 * vec
                    if (temp_vec[0] * direction_vec[0] + temp_vec[1] * direction_vec[1] + temp_vec[2] * direction_vec[2]) > 0:
                        value = (dist - r_e) * (vec / dist)
                        edge_forces[index][0] = interaction_strength[cell_1_type + cell_2_type] * value
                        edge_forces[index][1] = -1 * interaction_strength[cell_1_type + cell_2_type] * value
                    else:
                        value = (dist - r_e) * (vec / dist)
                        edge_forces[index][0] = -1 * U_gc_rep * value
                        edge_forces[index][1] = U_gc_rep * value
                        # edge_forces[index][0] = 0
                        # edge_forces[index][1] = 0
                else:
                    # get value prior to applying type specific adhesion const
                    value = (dist - r_e) * (vec / dist)
                    edge_forces[index][0] = interaction_strength[cell_1_type + cell_2_type] * value
                    edge_forces[index][1] = -1 * interaction_strength[cell_1_type + cell_2_type] * value
    return edge_forces


@jit(nopython=True, parallel=True)
def get_gravity_forces(number_cells, locations, center, well_rad, net_forces, grav=2):
    for index in range(number_cells):
        new_loc = locations[index] - center
        new_loc_sum = new_loc[0] ** 2 + new_loc[1] ** 2
        net_forces[index] = -grav * (new_loc / well_rad) * new_loc_sum ** (1 / 2)
    return net_forces


def ref_angle_2_cart(rho, theta, phi, cell, center):
    # convert degrees to radians:
    theta = theta * (np.pi/180)
    phi = phi * (np.pi/180)
    x = rho * np.cos(theta) * np.sin(phi)
    y = rho * np.sin(theta) * np.sin(phi)
    z = rho * np.cos(phi)
    return np.hstack((x, y, z)) + (cell - center)

def cart2angle(vec):
    r = (vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2) ** (1/2)
    theta = np.arctan2(vec[1], vec[0]) * 180/np.pi
    phi = np.arccos(vec[2]/r) * 180/np.pi
    return theta, phi

@jit(nopython=True)
def convert_edge_forces(number_edges, edges, edge_forces, neighbor_forces):
    for index in range(number_edges):
        # get indices of cells in edge
        cell_1 = edges[index][0]
        cell_2 = edges[index][1]

        neighbor_forces[cell_1] += edge_forces[index][0]
        neighbor_forces[cell_2] += edge_forces[index][1]

    return neighbor_forces


def set_div_thresh(cell_type):
    """ Specify division threshold value for a particular cell.

        Distribution of cell division thresholds modeled by a shifted gamma distribution
        from Stukalin et. al., RSIF 2013
    """
    # parameters for gamma distribution
    alpha, a_0, beta = 12.5, 10.4, 0.72

    # based on cell type return division threshold in seconds
    if cell_type > 0:
        alpha, a_0, beta = 12.5, 10.4, 0.72
        hours = r.gammavariate(alpha, beta) + a_0
        # CHO cell time < HEK cell time
    else:
        alpha, a_0, beta = 10, 10.4, 0.72
        hours = r.gammavariate(alpha, beta) + a_0

    return hours * 10


def seed_cells(num_agents, center, radius):
    theta = 2 * np.pi * np.random.rand(num_agents).reshape(num_agents, 1)
    rad = radius * np.sqrt(np.random.rand(num_agents)).reshape(num_agents, 1)
    x = rad * np.cos(theta) + center[0]
    y = rad * np.sin(theta) + center[1]
    z = np.zeros((num_agents, 1)) + center[2]
    locations = np.hstack((x, y, z))
    return locations

# cell angle in degrees
def seed_angles(num_agents):
    theta = 360 * np.random.rand(num_agents).reshape(num_agents, 1)
    return theta

class CPSimulation(Simulation):
    """ This class inherits the Simulation class allowing it to run a
        simulation with the proper functionality.
    """
    def __init__(self, model_params):
        # initialize the Simulation object
        Simulation.__init__(self)

        default_params = {
            'cuda': False,
            'well_rad': 50,
            'output_values': True,
            'output_images': True,
            'image_quality': 500,
            'video_quality': 500,
            'fps': 10,
            'replication_type': None,
            'initial_seed_ratio': 0.1,
            'cell_interaction_rad': 3.2,
            'size': [1, 1, 1],
            'cell_rad': 0.5,
            'gel_rad': 0.5,
            'sub_ts': 10
        }
        self.model_parameters(default_params)
        # read parameters from YAML file and add them to instance variables
        self.model_parameters(model_params)

        # cell and gel colors
        self.cell_color = np.array([255, 50, 50], dtype=int) # red
        self.gel_color = np.array([50, 50, 255], dtype=int) # blue

        # Simulation space parameters
        self.initial_seed_rad = self.well_rad * self.initial_seed_ratio
        self.dim = np.asarray(self.size)
        self.size = self.dim * self.well_rad
        self.center = np.array([self.size[0]/2, self.size[1]/2, 0])

    def setup(self):
        """ Overrides the setup() method from the Simulation class.
        """
        # determine the number of agents for each cell type and
        # add agents to the simulation
        self.add_agents(self.num_cells, agent_type="CELL")
        self.add_agents(self.num_gels, agent_type="GEL")

        # indicate agent arrays and create the arrays with initial conditions
        self.indicate_arrays("locations", "radii", "colors", "cell_type", "division_set", "div_thresh")

        # generate random locations for cells
        self.locations = seed_cells(self.number_agents, self.center, self.initial_seed_rad)
        self.radii = self.agent_array(initial={'CELL': lambda: self.cell_rad, 'GEL': lambda: self.gel_rad})

        # Define cell types, 2 is ABA, 1 is DOX, 0 is non-cadherin expressing cho cells
        self.cell_type = self.agent_array(dtype=int, initial={"CELL": lambda: 1, "GEL": lambda: 0})
        self.colors = self.agent_array(dtype=int, vector=3, initial={"CELL": lambda: self.cell_color, "GEL": lambda: self.gel_color})
        # self.cell_angle_phi = seed_angles(self.number_agents)/2
        # self.cell_angle_theta = seed_angles(self.number_agents)
        self.cell_angle_phi = np.zeros((self.number_agents, 1))
        self.cell_angle_theta = np.zeros((self.number_agents, 1))

        # setting division times (in seconds):
        self.div_thresh = self.agent_array(initial={"CELL": lambda: set_div_thresh(0), "GEL": lambda: set_div_thresh(0)})
        self.division_set = self.agent_array(initial={"CELL": lambda: 17 * r.random(), "GEL": lambda: 17 * r.random()})


        #indicate and create graphs for identifying neighbors
        self.indicate_graphs("neighbor_graph")
        self.neighbor_graph = self.agent_graph()

        # record initial values
        self.step_values()
        self.step_image()

    def step(self):
        """ Overrides the step() method from the Simulation class.
        """
        # increase division counter and determine if any cells are dividing
        self.reproduce(1)
        # preform subset force calculations
        for i in range(self.sub_ts):
            # get all neighbors within threshold (1.6 * diameter)
            self.get_neighbors(self.neighbor_graph, self.cell_interaction_rad * self.cell_rad)
            # move the cells and track total repulsion vs adhesion forces
            self.move_parallel()
        # get the following data. We can generate images at each time step, but right now that is not needed.

        # add/remove agents from the simulation
        self.update_populations()
        self.update_angles()
        self.step_values()
        self.step_image()
        self.temp()
        self.data()

    def end(self):
        """ Overrides the end() method from the Simulation class.
        """
        self.step_values()
        self.step_image()
        self.create_video()

    @record_time
    def update_populations(self):
        """ Adds/removes agents to/from the simulation by adding/removing
            indices from the cell arrays and any graphs.
        """
        # get indices of hatching/dying agents with Boolean mask
        add_indices = np.arange(self.number_agents)[self.hatching]
        remove_indices = np.arange(self.number_agents)[self.removing]

        # count how many added/removed agents
        num_added = len(add_indices)
        num_removed = len(remove_indices)
        # go through each agent array name
        for name in self.array_names:
            # copy the indices of the agent array data for the hatching agents
            copies = self.__dict__[name][add_indices]

            # add indices to the arrays
            self.__dict__[name] = np.concatenate((self.__dict__[name], copies), axis=0)

            # if locations array
            if name == "locations":
                # go through the number of cells added
                for i in range(num_added):
                    # get mother and daughter indices
                    mother = add_indices[i]
                    daughter = self.number_agents + i

                    # move distance of radius in random direction
                    vec = self.radii[i] * self.random_vector()
                    self.__dict__[name][mother] += vec
                    self.__dict__[name][daughter] -= vec

            # reset division time
            if name == "division_set":
                # go through the number of cells added
                for i in range(num_added):
                    # get mother and daughter indices
                    mother = add_indices[i]
                    daughter = self.number_agents + i

                    # set division counter to zero
                    self.__dict__[name][mother] = 0
                    self.__dict__[name][daughter] = 0

            # set new division threshold
            if name == "division_threshold":
                # go through the number of cells added
                for i in range(num_added):
                    # get daughter index
                    daughter = self.number_agents + i

                    # set division threshold based on cell type
                    self.__dict__[name][daughter] = set_div_thresh(self.cell_type[daughter])

            # remove indices from the arrays
            self.__dict__[name] = np.delete(self.__dict__[name], remove_indices, axis=0)

        # go through each graph name
        for graph_name in self.graph_names:
            # add/remove vertices from the graph
            self.__dict__[graph_name].add_vertices(num_added)
            self.__dict__[graph_name].delete_vertices(remove_indices)

        # change total number of agents and print info to terminal
        self.number_agents += num_added
        # print("\tAdded " + str(num_added) + " agents")
        # print("\tRemoved " + str(num_removed) + " agents")

        # clear the hatching/removing arrays for the next step
        self.hatching[:] = False
        self.removing[:] = False

    @record_time
    def move_parallel(self):
        edges = np.asarray(self.neighbor_graph.get_edgelist())
        num_edges = len(edges)
        edge_forces = np.zeros((num_edges, 2, 3))
        neighbor_forces = np.zeros((self.number_agents, 3))
        self.direction_vectors = ref_angle_2_cart(1, self.cell_angle_theta, self.cell_angle_phi, self.locations, self.center)
        # get adhesive/repulsive forces from neighbors and gravity forces
        edge_forces = get_neighbor_forces(num_edges, edges, edge_forces, self.locations, self.center,
                                          self.direction_vectors, self.cell_type, self.cell_rad,
                                          U_gg=self.U_gg, U_cc=self.U_cc, U_gc=self.U_gc, U_gc_rep=self.U_gc_rep)
        neighbor_forces = convert_edge_forces(num_edges, edges, edge_forces, neighbor_forces)
        noise_vector = np.ones((self.number_agents, 3)) * self.alpha * (2 * np.random.rand(self.number_agents, 3) - 1)
        neighbor_forces = neighbor_forces + noise_vector
        if self.gravity > 0:
            net_forces = np.zeros((self.number_agents, 3))
            gravity_forces = get_gravity_forces(self.number_agents, self.locations, self.center, 325,
                                                net_forces, grav=self.gravity)
            neighbor_forces = neighbor_forces + gravity_forces
        for i in range(self.number_agents):
            vec = neighbor_forces[i]
            sum = vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2
            if sum != 0:
                neighbor_forces[i] = neighbor_forces[i] / (sum ** (1/2))
            else:
                neighbor_forces[i] = 0
        # update locations based on forces
        self.locations += 2 * self.velocity * self.cell_rad * neighbor_forces
        # check that the new location is within the space, otherwise use boundary values
        self.locations = np.where(self.locations > self.well_rad, self.well_rad, self.locations)
        self.locations = np.where(self.locations < 0, 0, self.locations)

    @record_time
    def reproduce(self, ts):
        """ If the agent meets criteria, hatch a new agent.
        """
        # increase division counter by time step for all agents
        #go through all agents marking for division if over the threshold
        if self.replication_type == 'Default':
            self.division_set += ts
            self.hatching = self.division_set > self.div_thresh
            return
        if self.replication_type == 'None':
            return

    @classmethod
    def simulation_mode_0(cls, name, output_dir, model_params):
        """ Creates a new brand new simulation and runs it through
            all defined steps.
        """
        # make simulation instance, update name, and add paths
        sim = cls(model_params)
        sim.name = name
        sim.set_paths(output_dir)

        # set up the simulation agents and run the simulation
        sim.full_setup()
        sim.run_simulation()

    def save_params(self, path):
        """ Add the instance variables to the Simulation object based
            on the keys and values from a YAML file.
        """
        # load the dictionary
        params = template_params(path)

        # iterate through the keys adding each instance variable
        with open(self.main_path + "parameters.txt", "w") as parameters:
            for key in list(params.keys()):
                parameters.write(f"{key}: {params[key]}\n")
        parameters.close()

    def update_angles(self):
        """ Update the angles of the cells.
        """
        phi_update = np.zeros((self.number_agents, 1))
        theta_update = np.zeros((self.number_agents, 1))
        adjacency_matrix = self.neighbor_graph.get_adjacency()
        for index in range(self.number_agents):
            # checking to see if the agent is a cell.
            phi = 0
            theta = 0
            if self.cell_type[index] == 1:
                neighbors = np.nonzero(adjacency_matrix[index])[0]
                if len(neighbors) > 0:
                    cell_loc = self.locations[index] - self.center
                    for agent in neighbors:
                        if self.cell_type[agent] == 0:
                            gel_loc = self.locations[agent] - self.center
                            dist = cell_loc - gel_loc
                            mag = (dist[0] ** 2 + dist[1] ** 2 + dist[2] ** 2) ** (1/2)
                            gel_theta, gel_phi = cart2angle(dist)
                            phi = phi + (gel_phi - self.cell_angle_phi[index])/mag
                            theta = theta + (gel_theta - self.cell_angle_theta[index])/mag
            if abs(phi) > self.rotation_threshold:
                phi_update[index] = phi/abs(phi)
            else:
                phi_update[index] = 0
            if abs(theta) > self.rotation_threshold:
                theta_update[index] = theta/abs(theta)
            else:
                theta_update[index] = 0
        self.cell_angle_phi = self.cell_angle_phi + 10 * phi_update #+ 5 * (2 * np.random.rand(self.number_agents, 1) - 1)
        self.cell_angle_theta = self.cell_angle_theta + 10 * theta_update #+ 5 * (2 * np.random.rand(self.number_agents, 1) - 1)

    @classmethod
    def start_sweep(cls, output_dir, model_params, name):
        """ Configures/runs the model based on the specified
            simulation mode.
        """
        # check that the output directory exists and get the name/mode for the simulation
        output_dir = backend.check_output_dir(output_dir)

        name = backend.check_existing(name, output_dir, new_simulation=True)
        cls.simulation_mode_0(name, output_dir, model_params)

def parameter_sweep_abm(par, directory, GC, CC, GG, GCrep, num_cell, num_gel, gravity, velocity, r_thresh, final_ts=60):
    """ Run model with specified parameters
        :param par: simulation number
        :param directory: Location of model outputs. A folder titled 'outputs' is required
        :param GC: Adhesion value for G-C cell interactions
        :param CC: Adhesion value for C-C cell interactions
        :param GG: Adhesion value for G-G cell interactions
        :param GCrep: Repulsion value for C-G cell interactions
        :param num_cell: initial ratio of cell agents.
        :param num_gel: initial ratio of gel agents.
        :param final_ts: Final timestep. 60 ts = 96h, 45 ts = 72h
        :type par int
        :type directory: String
        :type GC: float
        :type CC: float
        :type GG: float
        :type GC: float
        :type num_cell: float
        :type num_gel: float
        :type final_ts: int
    """
    agent_params = {
        'num_cells': num_cell,
        'num_gels': num_gel,
        'end_step': final_ts,
        'gravity': gravity, #5
        'velocity': velocity, #0.3
        'alpha': 10,
        'U_gg': GG,
        'U_cc': CC,
        'U_gc': GC,
        'U_gc_rep': GCrep,
        'rotation_threshold': r_thresh #10
    }
    name = f'ugc{GC}_ucc{CC}_ugg{GG}_ugcrep{GCrep}_gel{num_gel}_cell{num_cell}'
    sim = CPSimulation(agent_params)
    sim.start_sweep(directory + '/outputs', agent_params, name)
    return par, sim.image_quality, sim.image_quality, 3, final_ts/sim.sub_ts


if __name__ == "__main__":
    directory = "/Users/andrew/PycharmProjects/cell_polarity_organoid/"
    num_cell = [900, 800, 700, 600, 500, 400, 300, 200, 100]
    num_gel = [100, 200, 300, 400, 500, 600, 700, 800, 900]
    for i in range(len(num_gel)):
        parameter_sweep_abm(0, directory, 30, 5, 1, 70, num_cell[i], num_gel[i], 2, 0.3, 10, final_ts=240)
    # parameter_sweep_abm(0, directory, 30, 5, 40, 200, 200, 2, 0.3, 10, final_ts=240)