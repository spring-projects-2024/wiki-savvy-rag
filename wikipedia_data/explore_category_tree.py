import tqdm
import json
import pandas as pd
from collections import defaultdict


def depth_first(data, key, visited, depth, max_depth):
    if depth > max_depth:
        return
    visited.add(key)
    if key in data:
        for child in data[key]:
            if child not in visited:
                depth_first(data, child, visited, depth + 1, max_depth)
    return visited


tree_path = "../../archive/children_cats.csv"
df = pd.read_csv("../../archive/children_cats.csv")

parent_to_children = defaultdict(list)
child_to_parents = defaultdict(list)
for index, row in tqdm.tqdm(df.iterrows()):
    parent = row["parent"].lower()
    children = [x.lower() for x in row["children"].split()]
    parent_to_children[parent].extend(children)
    for child in children:
        child_to_parents[child].append(parent)

parent_to_children = {k: list(set(v)) for k, v in parent_to_children.items()}
child_to_parents = {k: list(set(v)) for k, v in child_to_parents.items()}

max_depth = 3
roots = ['Formal_sciences', 'Mathematics', 'Mathematics_education', 'Equations', 'Heuristics', 'Measurement', 'Numbers',
         'Proofs', 'Theorems', 'Fields_of_mathematics', 'Arithmetic', 'Algebra', 'Geometry', 'Trigonometry',
         'Mathematical_analysis', 'Calculus', 'Logic', 'Deductive_reasoning', 'Inductive_reasoning', 'History_of_logic',
         'Fallacies', 'Metalogic', 'Philosophy_of_logic', 'Mathematical_sciences', 'Computational_science',
         'Operations_research', 'Theoretical_physics', 'Statistics', 'Analysis_of_variance', 'Bayesian_statistics',
         'Categorical_data', 'Covariance_and_correlation', 'Data_analysis', 'Decision_theory', 'Design_of_experiments',
         'Logic_and_statistics', 'Multivariate_statistics', 'Non-parametric_statistics', 'Parametric_statistics',
         'Regression_analysis', 'Sampling', 'Statistical_theory', 'Stochastic_processes', 'Summary_statistics',
         'Survival_analysis', 'Time_series', 'Science', 'Natural_sciences', 'Nature', 'Biology', 'Botany', 'Ecology',
         'Health_sciences', 'Medicine', 'Neuroscience', 'Zoology', 'Earth_sciences', 'Atmospheric_sciences',
         'Geography', 'Geology', 'Geophysics', 'Oceanography', 'Nature', 'Animals', 'Environment', 'Humans', 'Life',
         'Natural_resources', 'Plants', 'Pollution', 'Physical_sciences', 'Astronomy', 'Chemistry', 'Climate',
         'Physics', 'Space', 'Universe', 'Scientific_method', 'Scientists', 'Technology', 'Applied_sciences',
         'Agriculture', 'Agronomy', 'Architecture', 'Automation', 'Biotechnology', 'Cartography',
         'Chemical_engineering', 'Communication', 'Media_studies', 'Telecommunications', 'Construction',
         'Control_theory', 'Design', 'Digital_divide', 'Earthquake_engineering', 'Energy', 'Ergonomics', 'Firefighting',
         'Fire_prevention', 'Forensic_science', 'Forestry', 'Industry', 'Information_science', 'Internet', 'Management',
         'Manufacturing', 'Marketing', 'Medicine', 'Unsolved_problems_in_neuroscience', 'Metalworking',
         'Microtechnology', 'Military_science', 'Mining', 'Nanotechnology', 'Nuclear_technology', 'Optics', 'Plumbing',
         'Robotics', 'Sound_technology', 'Technology_forecasting', 'Tools', 'Computing', 'Apps',
         'Artificial_intelligence', 'Classes_of_computers', 'Companies', 'Computer_architecture', 'Computer_model',
         'Computer_engineering', 'Computer_science', 'Computer_security', 'Computing_and_society', 'Data',
         'Embedded_systems', 'Free_software', 'Human–computer_interaction', 'Information_systems',
         'Information_technology', 'Internet', 'Mobile_web', 'Languages', 'Multimedia', 'Operating_systems',
         'Platforms', 'Product_lifecycle_management', 'Programming', 'Real-time_computing', 'Software',
         'Software_engineering', 'Unsolved_problems_in_computer_science', 'Electronics', 'Avionics', 'Circuits',
         'Companies', 'Connectors', 'Consumer_electronics', 'Digital_electronics', 'Digital_media',
         'Electrical_components', 'Electronic_design', 'Electronics_manufacturing', 'Embedded_systems',
         'Integrated_circuits', 'Microwave_technology', 'Molecular_electronics', 'Water_technology', 'Optoelectronics',
         'Quantum_electronics', 'Radio_electronics', 'Semiconductors', 'Signal_cables', 'Surveillance',
         'Telecommunications', 'Engineering', 'Aerospace_engineering', 'Bioengineering', 'Chemical_engineering',
         'Civil_engineering', 'Electrical_engineering', 'Environmental_engineering', 'Materials_science',
         'Mechanical_engineering', 'Nuclear_technology', 'Software_engineering', 'Structural_engineering',
         'Systems_engineering', 'Transport', 'By_country', 'Aviation', 'Cars', 'Cycling', 'Public_transport',
         'Rail_transport', 'Road_transport', 'Shipping', 'Spaceflight', 'Vehicles', 'Water_transport',
         'Technology_timelines']
selected = set()
for root in roots:
    visited = depth_first(parent_to_children, root.lower(), set(), 0, max_depth)
    selected.update(visited)

selected = list(selected)
selected.sort()
with open("selected_categories.json", "w") as f:
    json.dump(selected, f)