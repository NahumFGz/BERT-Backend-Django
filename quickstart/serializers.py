from django.contrib.auth.models import User, Group
from rest_framework import serializers
from quickstart.models import *

class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ['url', 'username', 'email', 'groups']


class GroupSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Group
        fields = ['url', 'name']

class PredictSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Predict
        fields = ['text','label_predict', 'label_probability']