from django.db import models
#from nlpmodule.tools.BertClassifier import BertClassifier
from nlpmodule.tools.Classifier import classifier_treatment_load

# Create your models here.
class Predict(models.Model):
    text = models.TextField()
    label_predict = models.TextField(null=True, blank=True)
    label_probability = models.DecimalField(null=True, blank=True, max_digits=5, decimal_places=4)

    
    def save(self, *args, **kwargs):

        try:
            self.label_predict, self.label_probability = classifier_treatment_load(self.text)
        except:
            self.label_predict, self.label_probability = '',0.0        
        
        super().save(*args, **kwargs)
        