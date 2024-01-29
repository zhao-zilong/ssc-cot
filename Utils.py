import pandas as pd
import numpy as np
import re
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import openai
from openai.error import OpenAIError 
from Trimaster100 import *

max_tokens_length = 12000
model_name = "gpt-3.5-turbo-16k"
YOUR_API_KEY =  "YOUR API KEY"
openai.api_key = YOUR_API_KEY

def thought_generator(question, intermediate_result=None, KG_information=None):
    try:
        if intermediate_result and KG_information:
            completion = openai.ChatCompletion.create(
                model=model_name,
                temperature=0.7,
                max_tokens=max_tokens_length,
                messages=[
                    {"role": "system", "content": "You are a very talented student who is currently competing in the International Mathematical Olympiad."},
                    {"role": "user", "content": question_template_IR_KG(question, intermediate_result, KG_information)}
                ],
                request_timeout=60,
            )
        
        elif not intermediate_result and KG_information:
            completion = openai.ChatCompletion.create(
                model=model_name,
                temperature=0.7,
                max_tokens=max_tokens_length,
                messages=[
                    {"role": "system", "content": "You are a very talented student who is currently competing in the International Mathematical Olympiad."},
                    {"role": "user", "content": question_template_KG(question, KG_information)}
                ],
                request_timeout=60,
            )
        
        elif intermediate_result and not KG_information:
            completion = openai.ChatCompletion.create(
                model=model_name,
                temperature=0.7,
                max_tokens=max_tokens_length,
                messages=[
                    {"role": "system", "content": "You are a very talented student who is currently competing in the International Mathematical Olympiad."},
                    {"role": "user", "content": advanced_question_template(question, intermediate_result)}
                ],
                request_timeout=60,
            )
        
        elif not intermediate_result and not KG_information:
            completion = openai.ChatCompletion.create(
                model=model_name,
                temperature=0.7,
                max_tokens=max_tokens_length,
                messages=[
                    {"role": "system", "content": "You are a very talented student who is currently competing in the International Mathematical Olympiad."},
                    {"role": "user", "content": question_template(question)}
                ],
                request_timeout=60,
            )
        
        return completion.choices[0].message['content']

    except OpenAIError as e:
        print("Timeout or API error occurred, retrying...")
        return thought_generator(question, intermediate_result, KG_information)
    
    
def single_result_extractor(thought):
    try:
        completion = openai.ChatCompletion.create(
            model=model_name,
            temperature=0.7,
            max_tokens=4000,
            messages=[
                {"role": "system", "content": "You are a very talented student who is very good at math and is attending the International Mathematical Olympiad now."},
                {"role": "user", "content": "given the reasoning: '{0}'. Output only the intermediate result (pure equation without extra text): ".format(thought)}
            ],
            request_timeout=60,  # Set your desired timeout
        )
        return completion.choices[0].message['content']

    except OpenAIError as e:  # Adjust the exception type as needed
        print("Timeout or API error occurred, retrying...")
        return single_result_extractor(thought)



def llm_state_evaluation(question, intermediate_result):
    try:
        completion = openai.ChatCompletion.create(
          model = model_name,
          temperature = 0,
          max_tokens = 4000,
          messages = [
            {"role": "system", "content": "You are a very talented student who is very good at math and is attending the International Mathematical Olympiad now."},
            {"role": "user", "content": llm_state_evaluation_template(question, intermediate_result)}
          ],
          request_timeout=60
        ) 
        return completion.choices[0].message['content']
    
    except OpenAIError as e:  # Adjust the exception type as needed
        print("Timeout or API error occurred, retrying...")
        return llm_state_evaluation(question, intermediate_result)
    
    
def state_evaluation_refinement(message):
    try:
        completion = openai.ChatCompletion.create(
          model = model_name,
          temperature = 0.7,
          max_tokens = 4000,
          messages = [
            {"role": "system", "content": "You are an assistant for your math professor, you are trying to summarize the result."},
            {"role": "user", "content": "Q: With the given text as follows, what is the concret number of the chosen intermediate result. Please only output the chosen arabic number: 'Based on the given intermediate results, the most promising intermediate result to solve the question is intermediate result 1: tan(100) + 4sin(100) = -tan(10) + 4sin(10). Therefore, the output is 1.' A. 1  Q. With the given text as follows, what is the concret number of the chosen intermediate result. Please only output the chosen arabic number: {0}. A: ".format(message)}
          ],
          request_timeout=60
        ) 
        return completion.choices[0].message['content']
    except OpenAIError as e:  # Adjust the exception type as needed
        print("Timeout or API error occurred, retrying...")
        return state_evaluation_refinement(message)
    
def state_correctness_evaluation(question, intermediate_result):
    try:
        completion = openai.ChatCompletion.create(
          model = model_name,
          temperature = 0,
          max_tokens = 4000,
          messages = [
            {"role": "system", "content": "You are a very talented student who is very good at math and is attending the International Mathematical Olympiad now."},
            {"role": "user", "content": state_correctness_verification_template(question, intermediate_result)}
          ],
          request_timeout=60
        ) 
        return completion.choices[0].message['content']
    except OpenAIError as e:  # Adjust the exception type as needed
        print("Timeout or API error occurred, retrying...")
        return state_correctness_evaluation(question, intermediate_result)
    
def state_correctness_refinement(message):
    try:
        completion = openai.ChatCompletion.create(
          model = model_name,
          temperature = 0.7,
          max_tokens = 4000,
          messages = [
            {"role": "system", "content": "You are an assistant for your math professor, you are trying to summarize the result."},
            {"role": "user", "content": "We have the inference as follows: {0}. According to the inference, please conclude a clear answer. Please only output yes or no.".format(message)}
          ],
          request_timeout=60
        ) 
        return completion.choices[0].message['content']
    
    except OpenAIError as e:  # Adjust the exception type as needed
        print("Timeout or API error occurred, retrying...")
        return state_correctness_refinement(message)

def extract_equations(text):
    '''
    Extract Exact intermediate results from reasoning text
    '''
    # Regular expression to match patterns like "1. " followed by any text up to the next such pattern or end of text
    pattern = r'\d+\. (.*?)((?=\d+\. )|$)'
    matches = re.findall(pattern, text, re.DOTALL)

    # Extracting only the equations from the matches
    equations = [match[0].strip() for match in matches]
    
    # Extracting only the equations from the matches
    equations_clean = [equation.strip(".") for equation in equations]
    return equations_clean

def select_overlapping_states(): 
    '''
    Allow manually select overlapping intermediate results by user.
    '''
    _input = input("Please enter the overlapping intermediate results' IDs (seperate with white space, use semicolon for different groups, for example: '1 3; 4 5; 2 6'): \n")
    pair_list = _input.split(";")
    result_list = []
    for pair in pair_list:
        result_list.append(pair.split())    
        
    return result_list


# Choose intermediate results for next round

def contains_elements_from_sublists(given_list, list_of_lists):
    """
    Checks if a given list contains at least one element from each sublist in a list of sublists.

    Parameters:
    - given_list (list): The list to check for presence of elements from the sublists.
    - list_of_lists (list of list): A list containing sublists, from which at least one element 
      from each sublist must be present in `given_list` for the function to return True.

    Returns:
    - bool: True if `given_list` contains at least one element from each sublist in `list_of_lists`,
      False otherwise.

    Example:
    >>> contains_elements_from_sublists([1, 2, 3], [[1, 4], [2, 5], [3, 6]])
    True
    >>> contains_elements_from_sublists([1, 2, 3], [[4, 5], [2, 6]])
    False
    """
    return all(any(element in given_list for element in sublist) for sublist in list_of_lists)

def unique_sublists(lst):
    """
    Removes duplicate sublists from a given list of sublists.

    This function takes a list of sublists (`lst`) and returns a new list containing only the unique sublists,
    preserving their original order in `lst`. A sublist is considered unique if there is no other sublist in `lst`
    with the exact same elements in the same order.

    Parameters:
    - lst (list of list): The input list containing sublists to be filtered for uniqueness.

    Returns:
    - list of list: A new list with only the unique sublists from the original list.

    Example:
    >>> unique_sublists([[1, 2], [3, 4], [1, 2], [5, 6]])
    [[1, 2], [3, 4], [5, 6]]
    """
    unique_sublists_set = set()
    result = []

    for sublist in lst:
        sublist_tuple = tuple(sublist)
        if sublist_tuple not in unique_sublists_set:
            unique_sublists_set.add(sublist_tuple)
            result.append(sublist)

    return result

def check_overlapping_groups(updated_buffer, overlapping_id_groups):
    '''
    Here overlapping_id_groups can only contain two sublists.
    In general, this function will check if there is one chain that contains node ID from two overlapping id groups,
    returns the most advanced id groups.
    '''
    # Flatten overlapping_id_groups into a single list
    all_overlapping_ids = [id_ for group in overlapping_id_groups for id_ in group]

    # Convert each sublist in updated_buffer into a list of IDs
    branches = [[node.id for node in sublist] for sublist in updated_buffer]

    # Initialize a list to hold the valid groups
    valid_groups = []

    # Analyze each branch
    for branch in branches:
        if contains_elements_from_sublists(branch, overlapping_id_groups):
            # Find the last ID in the branch that is in the flattened list of overlapping IDs
            last_id_in_branch = next((id_ for id_ in reversed(branch) if id_ in all_overlapping_ids), None)

            if last_id_in_branch is not None:
                # Determine which group the last ID belongs to
                for group in overlapping_id_groups:
                    if last_id_in_branch in group:
                        valid_groups.append(group)
                        break

    valid_groups = unique_sublists(valid_groups)

    
    if len(valid_groups) == 0: # If no valid groups were found in any branch, return all groups

        return overlapping_id_groups
    
    elif len(valid_groups) == 1:

        return valid_groups
    
    elif len(valid_groups) == 2:

        return valid_groups