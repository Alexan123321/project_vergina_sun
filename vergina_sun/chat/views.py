from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, logout

@login_required
def select_room(request):
    return render(request, 'chat/pages/select_room.html')

@login_required
def room(request, room_name):
    return render(request, 'chat/pages/room.html', {
        'room_name': room_name
    })