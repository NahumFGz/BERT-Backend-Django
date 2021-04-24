from django.urls import path
from .views import *

app_name = 'quickstart'
urlpatterns = [
    path('predict', view=PredictViewSet.as_view()),
]