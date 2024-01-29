from graphviz import Digraph
import os

# To visualize the flow chart, users need to load the Graphviz library.

os.environ["PATH"] += os.pathsep + 'PATH/TO/Graphviz/bin'


def solution_flow_visualization(json_data, save_path):

    intermediate_results = json_data['intermediate-results']

    # Create a Digraph object
    flowchart = Digraph('Flowchart', format='png')

    # Add the question as the starting node
    flowchart.node('0', f"Question:\n{json_data['question']}")

    # Add all nodes
    for result in intermediate_results:
        node_label = f"Step {result['step']}:\n{result['result']}"
        # results.append((result['result'], result['step'], result['score'], result['branch'], result['branch-level']))
        flowchart.node(str(result['step']), node_label)

    # Add all edges
    in_branch = False
    current_branch = None
    branch_start = None
    branch_end = []
    previous_branch_node = None
    for result in intermediate_results:    
        
        if result['branch'] != "None" and in_branch == False: # first time entering branch
            branch_start = str(result['step'] - 1) # "score" is actually the accumulated steps
            in_branch = True
            current_branch = result['branch']
            flowchart.edge(str(branch_start), str(result['step']))
            previous_branch_node = str(result['step'])

        elif result['branch'] != "None" and in_branch == True and current_branch == result['branch']: # within branches
            flowchart.edge(previous_branch_node, str(result['step']))
            previous_branch_node = str(result['step'])
            
        elif result['branch'] != "None" and current_branch != result['branch']: # switching branch
            flowchart.edge(branch_start, str(result['step']))
            branch_end.append(str(result["step"] - 1))
            previous_branch_node = str(result['step'])    
            current_branch = result['branch']

        elif result['branch'] == "None" and in_branch == True: # ending branches
            flowchart.edge(previous_branch_node, str(result['step']))
            for branch in branch_end:
                flowchart.edge(branch, str(result['step']))
                
            in_branch = False
            current_branch = None
            branch_start = None
            branch_end = []
            previous_branch_node = None
        else: # in main path
            flowchart.edge(str(result['step']-1), str(result['step']))
        
        
        
    # Save the flowchart to a file
    flowchart.render(save_path, format='png', cleanup=True)