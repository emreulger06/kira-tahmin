from django.urls import path
from .views import tahmin_view

urlpatterns = [
    path('', tahmin_view, name='tahmin_view'),  
]
