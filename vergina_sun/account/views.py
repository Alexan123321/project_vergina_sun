from django.shortcuts import render, redirect
from .forms import NewUserForm

from django.contrib.auth import login, logout
from django.contrib import messages

def login_request(request):
    return render(request, 'pages/login.html')

def logout_request(request):
    logout(request)
    #messages.info(request, "You have successfully logged out.") 
    return redirect('../')

def register_request(request):
    if request.method == "POST":
        form = NewUserForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            #messages.success(request, "Registration successful.")
            return redirect('account/login')
        #messages.error(request, "Unsuccesful registration. Invalid information.")
    form = NewUserForm
    return render(request=request, template_name='pages/register.html', context={"register_form":form})

def security_request(request):
    return redirect('/account/two_factor')
