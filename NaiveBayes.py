import math
import sys

class NaiveBayes: 
    # Initialize NaiveBayes with categories and a vocabulary set
    def __init__(self, Categories: list[str]):
        self.vocabulary: set = set()
        self.Categories: dict[str, Category] = {str(category): Category(category, self) for category in Categories}
        
    # Retrieve a specific category
    def __getitem__(self, key):
        return self.Categories.get(key)
    
    # Train model with a single text and category
    def fit(self, Text, strCategory):
        Category = self[strCategory]
        Category.Add_doc(Text)
    
    # Train model with multiple texts and categories
    def fit_all(self, TextList, CategoryList):
        for i in range(len(TextList)):
            self.fit(TextList[i], CategoryList[i])
    
    # Calculate total number of documents across all categories
    def totalDocs(self):
        Total = 0
        for Category in self.Categories:
            Total += self[Category].TotalDocCount
        return Total
    
    # Classify a text and return the highest scoring category
    def test(self, Text, use_log, smoothing):
        max_score = -sys.maxsize
        Total_Docs = self.totalDocs()
        for Category in self.Categories:
            Score = self[Category].GetScore(Text, Total_Docs, len(self.vocabulary), use_log, smoothing)
            if Score[1] > max_score:
                max_score = Score[1]
                Max_Category = Score
        return Max_Category[0]
    
    # Evaluate a single prediction against the true label
    def evaluate_single_prediction(self, Text, True_value):
        predicted_value = self.test(Text, True_value)[0]
        return True_value == predicted_value
    
    # Evaluate model accuracy over multiple texts
    def evaluate_predictions(self, TextList, LabelsList):
        total, correct = 0, 0
        for i in range(len(TextList)):
            if self.evaluate_single_prediction(TextList[i], LabelsList[i]):
                correct += 1
            total += 1
        return correct / total
            
class Category:
    # Initialize a Category with a name, a link to NaiveBayes, and counters
    def __init__(self, name: str, father):
        self.NaiveBayes: NaiveBayes = father
        self.name: str = name
        self.terms: dict = {}
        self.TotalDocCount: int = 0
        self.TotalTermCount: int = 0

    # Add or update term frequency in the category
    def Add_term(self, term):
        self.terms[term] = self.terms.get(term, 0) + 1
        self.TotalTermCount += 1
        self.NaiveBayes.vocabulary.add(term)

    # Add a document by updating terms and document count
    def Add_doc(self, text: str):
        Terms = text.lower().split()
        for term in Terms:
            self.Add_term(term)
        self.TotalDocCount += 1

    # Get the count of a term in the category
    def Get_term_count(self, term):
        return self.terms.get(term, 0)

    # Calculate score (probability) of a text belonging to this category
    def GetScore(self, Text: str, TotalDocs: int, TotalVocab: int, use_log=True, smoothing=True):
        if use_log:
            P_of_category = math.log10(self.TotalDocCount / TotalDocs)
            Terms = Text.lower().split()
            for term in Terms:
                term_count = self.Get_term_count(term)
                denominator = self.TotalTermCount
                
                if smoothing:
                    term_count += 1
                    denominator += TotalVocab
                
                
                P_of_category += math.log10((term_count / denominator) + 1e-100)
        else:
            P_of_category = self.TotalDocCount / TotalDocs
            Terms = Text.lower().split()
            for term in Terms:
                term_count = self.Get_term_count(term)
                denominator = self.TotalTermCount
                if smoothing:
                    term_count += 1
                    denominator += TotalVocab
                P_of_category *= term_count / denominator

        return [self.name, P_of_category]
