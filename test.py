from nlpmodule.tools.BertClassifier import BertClassifier
from nlpmodule.tools.Classifier import classifier_treatment_load

text   = 'She is gentle and kind, she never disrespected me'
a,b = classifier_treatment_load(text)

print(a)
print(b)