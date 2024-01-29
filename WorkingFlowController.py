import pandas as pd
import numpy as np
import re
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from Utils import *


# define the node for reasoning process
class Node:
    def __init__(self, 
                 _id,
                 _round,
                 value, 
                 isactive = True):
        self.id = _id
        self.round = _round
        self.value = value
        self.isactive = isactive
        self.children = []
    
    def set_isactive(self, _isactive):
        self.isactive = _isactive
        
    def add_child(self, child):
        # Add a child node to the list of children
        self.children.append(child) 

class WorkingFlowController():
    def __init__(self, 
                 breadth = 5,
                 deepth = 3,
                 questionjson = None,
                 knowledgegraph = None
                ):
        self.breadth = breadth
        self.deepth = deepth
        self.questionjson = questionjson
        self.question = questionjson['question']
        self.original_form = None
        try:
            self.original_form = questionjson['original_form']
        except:
            self.original_form = None
        self.thought_collector = []
        self.intermediate_result_list = []
        self.tree_list = []
        self.chosen_state = []
        self.KG = knowledgegraph
        self.hint = []
        self.num_of_node = 0

    def set_nodes_inactive_with_overlap_ids(self, overlapping_node_list):

        overlapping_ids = list(set(sum(overlapping_node_list, [])))
        overlapping_ids = [int(i) for i in overlapping_ids]

        # Update Node statuses based on overlapping IDs
        for node_id in overlapping_ids:
            for subtree in self.tree_list: # check for all the chain, even previous round
                for sublist in subtree:
                    for node in sublist:
                        if node.id == node_id:
                            index_in_sublist = sublist.index(node)
                            # Set all previous nodes in the same sublist as inactive
                            for i in range(index_in_sublist + 1):
                                sublist[i].set_isactive(False)       


    def display_all_node(self):
        if self.tree_list != []:
            for subtree in self.tree_list:
                for sublist in subtree:
                    for node in sublist:
                        print(f"Node ID: {node.id}, Round: {node.round}, Value: {node.value}, IsActive: {node.isactive}")  
        
    def display_active_node(self):
        if self.tree_list != []:
            for subtree in self.tree_list:
                for sublist in subtree:
                    for node in sublist:
                        if node.isactive == True:
                            print(f"Node ID: {node.id}, Round: {node.round}, Value: {node.value}, IsActive: {node.isactive}")            

                            
    def state_selection(self):
        
        current_nodes = self.tree_list[-1]
        intermediate_results = ""
        for idx, nodes in enumerate(current_nodes):
            intermediate_results = intermediate_results + str(idx+1) + ". "+nodes[-1].value+" "
            
        evaluation_ = llm_state_evaluation(self.question, intermediate_results)
        
        # Here we allow LLMs to summarize the choice twice just in case it encounters errors
        try:
            chosen_state = int(state_evaluation_refinement(evaluation_))
            return chosen_state
        except:
            try:
                chosen_state = int(state_evaluation_refinement(evaluation_))
                return chosen_state
            except:
                print("The LLMs cannot choose one intermediate results among all.")
                return None
        
                                 
                            
    def evaluate_group_list(self, group_list = []):
        '''
        Here group list can be more than 2 sublists. It will pair-wise check node id group priority between each sublist pair 
        in group_list
        '''

        if group_list == []: # there is no overlapping states
            # in that case, we will only let LLM to choose one state for next round
            chosen_state = self.state_selection() # the chosen_state is actually the branch number, we always use the last state of one branch for LLMs to choose
            # if there is no chosen state, we do not update self.chosen_state, hint will be based on previous chosen states
            if chosen_state:
                subtree = self.tree_list[-1]
                chosen_node_id = subtree[chosen_state-1][-1].id # chosen_state-1 because our chosen_state starts from 1
                self.chosen_state.append([[chosen_node_id]])
        else:
            # List to keep track of groups to be removed
            groups_to_remove = []

            for i in range(len(group_list)):
                if group_list[i] in groups_to_remove:
                    continue  # Skip if this group is marked for removal

                for j in range(i + 1, len(group_list)):
                    if group_list[j] in groups_to_remove:
                        continue  # Skip if the paired group is marked for removal

                    paired_group = group_list[j]

                    # here the check should happen in every chain-of-thought
                    flat_list = [item for sublist in self.tree_list for item in sublist]
                    result = check_overlapping_groups(flat_list, [group_list[i], paired_group])
                    
                    if len(result) == 1:
                        if result[0] == group_list[i]:
                            groups_to_remove.append(paired_group)
                        elif result[0] == paired_group:
                            groups_to_remove.append(group_list[i])
                            break  # No need to check further pairs for this group
                    elif len(result) == 0:
                        groups_to_remove.append(paired_group)
                        groups_to_remove.append(group_list[i])
                        break
            # Create a new list excluding the removed groups
            final_group_list = [group for group in group_list if group not in groups_to_remove]

            self.chosen_state.append(final_group_list)

    def fetch_reasoning_chain(self, nodeid):
        result_list = []
        for tree in self.tree_list:
            for chain in tree:
                for _id, node in enumerate(chain):
                    if nodeid == node.id:
                        for i in range(_id+1):
                            result_list.append(chain[i].value)
        result_string =""
        for idx, var in enumerate(result_list):
            result_string = result_string + str(idx+1) + ". " + var + " "
        return result_string
    
    def get_first_node_values(self):
        '''
        extract unique node value for each id group, then combine them as new input for next round
        '''
        if len(self.chosen_state) == 0: # first round and no chosen state.
            self.intermediate_result_list.append(None)  
            return        
        
        # Flatten the updated_buffer to easily access nodes by ID
        all_nodes = [node for sublist in self.tree_list[-1] for node in sublist]
        node_id_to_value = {node.id: node.value for node in all_nodes}

        # Retrieve the first value from each group in final_groups
        first_values = []
        for group in self.chosen_state[-1]:
            first_id = None
            if len(group) != 0:
                first_id = group[0]  # Get the first ID in the group
            first_value = node_id_to_value.get(first_id)  # Get the corresponding value
            
            
            if first_value is not None:
                # Verify the correctness of intermediate results
                reasoning_chain_result = self.fetch_reasoning_chain(first_id)
                print("node id and chain value: ", first_id, reasoning_chain_result)
                # iscorrect = True
                # self.verify_correctness_intermediate_result(reasoning_chain_result)
                iscorrect = self.verify_correctness_intermediate_result(first_value)
                if iscorrect:
                    first_values.append(first_value)
                    print("Following intermediate result will be used in next round inference: ", first_value)
                    
                # We put all nodes in the corresponding chain of thought as inactive since there are errors within it.
                else:
                    for node_id in group:
                        for sublist in self.tree_list[-1]:
                            for node in sublist:
                                if node.id == node_id:
                                    # Set all nodes in the sublist as inactive
                                    for i in range(len(sublist)):
                                        sublist[i].set_isactive(False)  

        intermediate_results = ""
        for idx, val in enumerate(first_values):
            # intermediate_results are the list of chosen intermediate results that will be used for next round exploration.           
            intermediate_results = intermediate_results + str(idx+1) + ". "+val+" "

        if intermediate_results == "":
            self.intermediate_result_list.append(None)  
        else:
            self.intermediate_result_list.append(intermediate_results)
       

    def last_non_none_element(self):
        """
        Returns the last non-None element in the self.intermediate_result_list.
        Returns None if all elements are None or the list is empty.
        """
        for element in reversed(self.intermediate_result_list):
            if element is not None:
                return element
        return None
    
    def create_node_list(self, thought_chain_buffer, round_number):
        # Step 1: Create all Node objects
        updated_buffer = []
        node_id = self.num_of_node  # Starting id, we need this parameter because start from second round, the node id does not start from 0
        round_number = round_number  # Assuming a fixed round number for simplicity

        for sublist in thought_chain_buffer:
            new_sublist = []
            for node_value in sublist:
                node = Node(node_id, round_number, node_value, True)
                new_sublist.append(node)
                node_id += 1
            updated_buffer.append(new_sublist)

        # always update num_of_node after initializing node list
        self.num_of_node = self.num_of_node + len(list(chain.from_iterable(thought_chain_buffer)))
        print("num_of_node: ", self.num_of_node)
        
        return updated_buffer
        

        
    def active_node_value_similarity(self):
        '''
        This function can only verify pure text similarity, not semantically check the intermediate result similarity
        Transform the documents into TF-IDF vectors and then compute the cosine similarity between them
        '''
        text_elements = []
        node_ids = []
        if self.tree_list != []:
            for subtree in self.tree_list:
                for sublist in subtree:
                    for node in sublist:
                        if node.isactive == True:
                            text_elements.append(node.value) 
                            node_ids.append(node.id)

        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer()

        # Transform the text elements into TF-IDF vectors
        tfidf_matrix = vectorizer.fit_transform(text_elements)

        # Calculate cosine similarity between all pairs of elements
        similarities = cosine_similarity(tfidf_matrix)

        print("\n\n\nAccording to our function, following nodes can express similar meaning.")
        # Find and print pairs of elements with similarity above the threshold
        for i in range(len(text_elements)):
            for j in range(i + 1, len(text_elements)):
                if similarities[i][j] >= 0.9999:
                    print(f"Node {node_ids[i]} and {node_ids[j]} express the same meaning.")
        print("\n\n\n")

    def merge_similar_node_pairs(self, similar_node_pairs):
        merged_groups = []

        for pair in similar_node_pairs:
            found = False
            for group in merged_groups:
                if pair[0] in group or pair[1] in group:
                    group.update(pair)
                    found = True
                    break

            if not found:
                merged_groups.append(set(pair))

        # Convert sets to lists
        result = [list(group) for group in merged_groups]

        return result

    
    def active_node_value_similarity_auto(self, similarity_threshold=0.999):
        '''
        This function can only verify pure text similarity, not semantically check the intermediate result similarity
        Transform the documents into TF-IDF vectors and then compute the cosine similarity between them
        '''
        text_elements = []
        node_ids = []
        similar_node_pairs = []
        
        if self.tree_list != []:
            for subtree in self.tree_list:
                for sublist in subtree:
                    for node in sublist:
                        if node.isactive == True:
                            text_elements.append(node.value) 
                            node_ids.append(node.id)

        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer()

        # Transform the text elements into TF-IDF vectors
        tfidf_matrix = vectorizer.fit_transform(text_elements)

        # Calculate cosine similarity between all pairs of elements
        similarities = cosine_similarity(tfidf_matrix)

        print("\n\n\nAccording to our function, following nodes can express similar meaning.")
        # Find and print pairs of elements with similarity above the threshold
        
        # Create a dictionary to store connected nodes
        connected_pairs_set = set()
        
        
        for i in range(len(text_elements)):
            for j in range(i + 1, len(text_elements)):
                if similarities[i][j] >= similarity_threshold:
                    # Check if the pair is not already in the set
                    if (node_ids[i], node_ids[j]) not in connected_pairs_set and (node_ids[j], node_ids[i]) not in connected_pairs_set:
                        connected_pairs_set.add((node_ids[i], node_ids[j]))
                        similar_node_pairs.append([node_ids[i], node_ids[j]])
                        # print(f"Node {node_ids[i]} and {node_ids[j]} express the same meaning.")
        
        merged_result = self.merge_similar_node_pairs(similar_node_pairs)
        print(merged_result)
        print("\n\n\n")
        return merged_result
    
    def save_intermediate_results(self, path = None):
        
        id_list = []
        round_list = []
        value_list = []
        isactive_list = []

        if self.tree_list != []:
            for subtree in self.tree_list:
                for sublist in subtree:
                    for node in sublist:
                        id_list.append(node.id)
                        round_list.append(node.round)
                        value_list.append(node.value)
                        isactive_list.append(node.isactive)

        df = pd.DataFrame(list(zip(id_list, round_list, value_list, isactive_list)), columns =['id','round','value','isactive'])
        if path:
            df.to_csv(path, index=False)
        else:
            df.to_csv("answer.csv", index=False)

            
    def verify_correctness_intermediate_result(self, result):
        result_ = state_correctness_refinement(state_correctness_evaluation(self.question, result))
        print("the intermediate result: ", result, " Does the intermediate result pass the verification: ", result_)
        if 'yes' in result_ or 'Yes' in result_:
            return True
        else:
            return False
            
    def similarity_between_generation_and_result(self):

        text_elements = []
        node_ids = []
        for element in self.questionjson['intermediate-results']:
            text_elements.append(element['result'])
            node_ids.append(element['step'])

        text_elements_generation = []
        node_ids_generation = []
        if self.tree_list != []:
            for subtree in self.tree_list:
                for sublist in subtree:
                    for node in sublist:
                        text_elements_generation.append(node.value)
                        node_ids_generation.append(node.id)

        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer()

        # Transform the text elements into TF-IDF vectors
        tfidf_matrix = vectorizer.fit_transform(text_elements+text_elements_generation)
        # tfidf_matrix_generation = vectorizer.fit_transform(text_elements_generation)

        # Calculate cosine similarity between all pairs of elements
        similarities = cosine_similarity(tfidf_matrix)

        print("\n\n\nAccording to our function, following nodes can express similar meaning.")
        # Find and print pairs of elements with similarity above the threshold
        for i in range(len(text_elements)):
            for j in range(len(text_elements), len(text_elements)+len(text_elements_generation)):
                if similarities[i][j] >= 0.9999:
                    print(f"ground true node {node_ids[i]} and generation node {node_ids_generation[j-len(text_elements)]} express the same meaning.")
                    print("ground true: ", text_elements[i])
                    print("generation: ", text_elements_generation[j-len(text_elements)])
        print("\n\n\n")
            
    def solve(self):
        print("The question is: ", self.question)
        for _round in range(self.deepth):
            thought_chain_buffer = []
            if _round == 0:
                hint = None
                if self.KG:
                    hint = self.KG.query_related_information(self.question)
                    self.hint.append(hint)
                    print("hint: ", hint)
                for nb in range(self.breadth):
                    print("========== generating round : ", _round, ", and No.: ", nb, " chain of thought.")
                    thought = thought_generator(self.question, intermediate_result = None, KG_information = hint)
                    self.thought_collector.append(thought)
                    extracted_equations = extract_equations(thought)
                    if extracted_equations == []:
                        thought_chain_buffer.append([single_result_extractor(thought)])
                    else:
                        thought_chain_buffer.append(extracted_equations)

                # If the intermediate result only contains the transformed (right) part, not the original (left) form. Then add it
                if self.original_form:
                    for answers in thought_chain_buffer:
                        for idx, answer in enumerate(answers):
                            if "=" not in answer:
                                answers[idx] = self.original_form + " = " + answer

                updated_nodes = self.create_node_list(thought_chain_buffer, _round)
                
                # tree_list records update_nodes for each round
                self.tree_list.append(updated_nodes)
                
                # Display all active node
                # self.display_active_node()
                
                # We can let user to judge the overlapping states using following line of code
                # overlapping_node_list = select_overlapping_states()
                overlapping_node_list = self.active_node_value_similarity_auto()


                
                if any(overlapping_node_list):
                    # transfer every element in overlapping_node_list to integer
                    overlapping_id_groups = [[int(element) for element in sublist] for sublist in overlapping_node_list]
                    self.evaluate_group_list(overlapping_id_groups)
                else:
                    self.evaluate_group_list()
                # first_values contains all the intermediate results
                self.get_first_node_values()
                
                if any(overlapping_node_list):
                    # Process the buffer and get the updated list with Node objects
                    self.set_nodes_inactive_with_overlap_ids(overlapping_node_list)
                else:
                    # If no overlapping node, use the LLM chosen one
                    if len(self.chosen_state) !=0:
                        self.set_nodes_inactive_with_overlap_ids(self.chosen_state[-1])
                    
                # Display the result in a readable format
                # self.display_all_node()

    
            if _round > 0:
            
                hint = None
                if self.KG and self.last_non_none_element():
                    hint = self.KG.query_related_information(self.last_non_none_element())
                    self.hint.append(hint)
                    print("hint: ", hint, self.last_non_none_element())
                    
                for nb in range(self.breadth):
                    print("========== generating round : ", _round, ", and number: ", nb, " chain of thought.")
                    print("intermediate result: ", self.last_non_none_element())
                    thought = thought_generator(self.question, self.last_non_none_element(), hint)
                    self.thought_collector.append(thought)
                    extracted_equations = extract_equations(thought)
                    if extracted_equations == []:
                        thought_chain_buffer.append([single_result_extractor(thought)])
                    else:
                        thought_chain_buffer.append(extracted_equations)

                # If the intermediate result only contains the transformed (right) part, not the original (left) form. Then add it
                if self.original_form:
                    for answers in thought_chain_buffer:
                        for idx, answer in enumerate(answers):
                            if "=" not in answer:
                                answers[idx] = self.original_form + " = " + answer
                 
                updated_nodes = self.create_node_list(thought_chain_buffer, _round)
                
                # tree_list records update_nodes for each round
                self.tree_list.append(updated_nodes)
                
                # Display all active node
                # self.display_active_node()
                

                
                if _round + 1 < self.deepth: # no need to select overlapping states for last round

                
                    # We can let user to judge the overlapping states using following line of code
                    # overlapping_node_list = select_overlapping_states()
                    
                    overlapping_node_list = self.active_node_value_similarity_auto()

                    if any(overlapping_node_list):
                        # transfer every element in overlapping_node_list to integer
                        overlapping_id_groups = [[int(element) for element in sublist] for sublist in overlapping_node_list]
                        self.evaluate_group_list(overlapping_id_groups)
                    else:
                        self.evaluate_group_list()
                    # first_values contains all the intermediate results
                    self.get_first_node_values()

                    if any(overlapping_node_list):
                        # Process the buffer and get the updated list with Node objects
                        self.set_nodes_inactive_with_overlap_ids(overlapping_node_list)
                    else:
                        # If no overlapping node, use the LLM chosen one
                        if len(self.chosen_state) !=0:
                            self.set_nodes_inactive_with_overlap_ids(self.chosen_state[-1])

                    # Display the result in a readable format
                    # self.display_all_node()
                            
                