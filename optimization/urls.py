from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name="index"),  # Homepage
    path('generate-road-network/', views.generate_road_network, name="generate_road_network"),
]
