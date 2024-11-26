import pandas as pd
import numpy as np
import re
import openai
import Levenshtein
from Utils import YOUR_API_KEY 


def find_keywords_in_text(text, keywords):
    """
    Find and return the keywords present in the provided text.
    
    Args:
    text (str): The text to search within.
    keywords (list): A list of keywords to search for.
    
    Returns:
    list: A list of keywords found in the text, empty if none are found.
    """
    # Convert the text to lower case for case-insensitive search
    text = text.lower()
    
    # List to store keywords found
    found_keywords = []
    
    # Check each keyword in the list
    for keyword in keywords:
        # Convert the keyword to lower case to match the text conversion
        if keyword.lower() in text:
            found_keywords.append(keyword)
    
    return found_keywords

def is_numeric(s):
    '''
    Determine if a given string is numeric
    '''
    if not s:
        return False  # Empty string is not considered numeric
    if s[0] == '-' and s[1:].isdigit():
        return True  # Negative number with digits only

    return s.isdigit()


def extract_info(expression):
    """
    Extract Trigonometric patterns and angles from a given expression.

    Args:
        expression (str): The input expression containing information about Trigonometric patterns and angles.

    Returns:
        tuple: A tuple containing two lists:
            - The first list contains the names of Trigonometric patterns.
            - The second list contains angle expressions.

    Examples:
        expression = "Trigonometric patterns: sin(A), sin(B), tan(A), tan(B), cos(A). Angle(s): 180+A, -A, 360+A, A+180, -A, 180-A."
        function_names, angles = extract_info(expression)
        print("Function Names:", function_names)
        print("Angle(s):", angles)
    """
    # Find the index positions of "Trigonometric patterns:" and "Angle(s):"
    functions_start = expression.find("Trigonometric pattern(s):")
    angles_start = expression.find("Angle(s):")

    if functions_start == -1 or angles_start == -1:
        return [], []  # Return empty lists if the keywords are not found

    # Extract the text between "Trigonometric pattern(s)(s)s:" and "Angle(s):"
    functions_text = expression[functions_start + len("Trigonometric pattern(s):"):angles_start].strip()
    angles_text = expression[angles_start + len("Angle(s):"):].strip()

    # Remove the trailing period (".") from function names and angle expressions
    functions_text = functions_text.rstrip('.')
    angles_text = angles_text.rstrip('.')

    # Split the text into individual function names
    function_names = [fn.strip() for fn in functions_text.split(',')]

    # Split the text into individual angles
    angles = [angle.strip() for angle in angles_text.split(',')]
    angles_clean = [angle.replace(" ", "") for angle in angles]

    return function_names, angles_clean


def find_most_similar_element(element, text_list, distance_threshold = 5, similarity_threshold=0.85):

    """
    Find the most similar element in a list of text to a given element based on Levenshtein distance and similarity.

    Args:
        element (str): The element to compare against the elements in the text list.
        text_list (list): A list of text elements to compare with the given element.
        distance_threshold (int, optional): The maximum Levenshtein distance allowed for a match. Default is 5.
        similarity_threshold (float, optional): The minimum similarity score (0 to 1) for a match. Default is 0.85.

    Returns:
        str or None: The most similar element from the text list that meets the similarity and distance thresholds.
                     Returns None if no sufficiently similar element is found.

    Examples:
        element = "example text"
        text_list = ["exampel txet", "some other text", "another example", "example tex"]
        most_similar = find_most_similar_element(element, text_list)
        print(most_similar)  # Output: "example tex"
    """

    most_similar_element = None
    highest_similarity = 0

    for text in text_list:
        distance = Levenshtein.distance(element, text)
        max_len = max(len(element), len(text))
        similarity = (1 - distance / max_len)

        if similarity > highest_similarity and distance < distance_threshold:
            highest_similarity = similarity
            most_similar_element = text

    if highest_similarity > similarity_threshold:
        return most_similar_element
    else:
        return None

def key_word_extraction_llm(query):
    openai.api_key = YOUR_API_KEY

    completion = openai.ChatCompletion.create(
      model = "gpt-4",
      temperature = 0.5,
      max_tokens = 4000,
      messages = [
        {"role": "system", "content": "You are a very talented student who is currently competing in the International Mathematical Olympiad."},
        {"role": "user", "content": "Q: For question: 'Simplify tan(100) + sin(10)cos(10) + (cot(20))^2 + sin(180+A)'. Extract trigonometric function and angles from the question. Be careful, for the pattern such as sin(10)cos(10), we should extract the trigonometric function as sin(A)cos(B), and for (cot(20))^2, the extracted trigonometric function should be both cot(A) and (cot(A))^2. There is no need to solve the problem, just provide the relevant information. A: Trigonometric pattern(s): tan(A), sin(A)cos(B), cot(A), (cot(A))^2, sin(A). Angle(s): 100, 10, 180+A. Q: For question: {0}. Extract trigonometric function and angles from the question. Be careful, for the pattern such as sin(10)cos(10), we should extract the trigonometric function as sin(A)cos(B), and for (cot(20))^2, the extracted trigonometric function should be both cot(A) and (cot(A))^2. There is no need to solve the problem, just provide the relevant information. A:".format(query)}
      ]
    )
    return completion.choices[0].message['content']

class KnowledgeGraph:
    def __init__(self,
                kg_path):
 
        """
        Initialize the KnowledgeGraph object.

        Parameters:
        kg_path (str): The file path to the knowledge graph CSV file.

        Attributes:
        kg_path (str): Stores the file path of the knowledge graph.
        df (DataFrame): A pandas DataFrame containing the knowledge graph data.
        """
    
        self.kg_path = kg_path
        self.df = pd.read_csv(self.kg_path)
    
    def load_KG(self):
        """
        Load and process the knowledge graph from the DataFrame.

        This method initializes the relation matrix based on the unique heads and tails in the DataFrame.
        It also populates the relation matrix with the corresponding actions.
        """
        
        self.unique_heads = list(set(self.df['head']))
        self.unique_tails = list(set(self.df['tail']))

        # Create an empty matrix filled with zeros
        num_heads = len(self.unique_heads)
        num_tails = len(self.unique_tails)
        self.relation_matrix = [[0 for _ in range(num_tails)] for _ in range(num_heads)]

        # Add relations to the matrix
        for idx in self.df.index:
            head_index = self.unique_heads.index(self.df['head'][idx])
            tail_index = self.unique_tails.index(self.df['tail'][idx])
            action = self.df['action'][idx]
            self.relation_matrix[head_index][tail_index] = action    

    def show_KG(self):
        """
        Display the knowledge graph.

        This method prints out all the non-zero relations in the knowledge graph.
        """
        
        num_heads = len(self.unique_heads)
        num_tails = len(self.unique_tails)
        for i in range(num_heads):
            for j in range(num_tails):
                if self.relation_matrix[i][j] != 0:
                    print(f"{self.unique_heads[i]} {self.relation_matrix[i][j]} {self.unique_tails[j]}")
            
    def search_by_head(self, head: str = "", action: str = None):
        """
        Search for tails related to a given head and optionally a specific action.

        Parameters:
        head (str): The head entity to search for.
        action (str, optional): The action to filter the results by.

        Returns:
        list: A list of tails related to the given head and action.
        """
        
        tail_list = []
        if head not in self.unique_heads:
            return tail_list
        head_index = self.unique_heads.index(head)
                
        for tail in self.unique_tails:
            tail_index = self.unique_tails.index(tail) 
            action = self.relation_matrix[head_index][tail_index]
            if action != 0:
                tail_list.append(self.unique_tails[tail_index])
                    
        return tail_list

    def add_element_KG(self, head="", action="",tail=""):
        """
        Add a new element to the knowledge graph.

        Parameters:
        head (str): The head entity of the new record.
        action (str): The action of the new record.
        tail (str): The tail entity of the new record.
        """
        
        
        if head != "" and action != "" and tail != "":
            self.df.loc[len(self.df.index)] = [head, action, tail]
            self.df.to_csv(self.kg_path, index=False)  
            self.df = pd.read_csv(self.kg_path)
            self.load_KG()
            print("Record added to Knowledge Graph!")
            
    def delete_element_KG(self, head="", action="", tail=""):
        """
        Delete an element from the knowledge graph.

        Parameters:
        head (str): The head entity of the record to be deleted.
        action (str): The action of the record to be deleted.
        tail (str): The tail entity of the record to be deleted. 
        """
        
        if head != "" and action != "" and tail != "":
            try: 
                temp_df = self.df[self.df['head'].isin([head])]
                temp_temp_df = temp_df[temp_df['action'].isin([action])]
                chosen_index = temp_temp_df[temp_temp_df['tail'].isin([tail])].index[0]
                self.df.drop(chosen_index, axis = 0, inplace = True)
                self.df.to_csv(self.kg_path, index=False)
                self.df = pd.read_csv(self.kg_path)
                self.load_KG()
                print("Record deleted from Knowledge Graph!")
            except:
                print("There is no record matching the provided head-action-tail content or permission denied by the system to modify the file.")
              
    def query_related_information(self, query):
        '''
        Query the knowledge graph for related information based on a given query.

        Args:
            query (str): The query to search for related information in the knowledge graph.

        Returns:
            list: A list of related information retrieved from the knowledge graph. Each item in the list
                provides details related to the query. If no relevant information is found, an empty list
                is returned.
        '''
        tail_result = []

        matching_key_words = find_keywords_in_text(query, self.unique_heads)
        for head in matching_key_words:
            tails = self.search_by_head(head)
            tail_result.append(tails)
             

        
        return tail_result