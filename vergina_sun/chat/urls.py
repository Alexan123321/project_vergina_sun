from django.urls import path

from . import views

urlpatterns = [
    path('', views.select_room, name='select_room'),
    path('<str:room_name>/', views.room, name='room'),
]