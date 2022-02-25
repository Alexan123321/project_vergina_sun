from django.conf.urls import include
from django.contrib import admin
from django.urls import path
from two_factor.urls import urlpatterns as tf_urls

from . import views

urlpatterns = [
    path('', views.homepage, name='homepage'),
    path('', include(tf_urls)),
    path('admin/', admin.site.urls),
    path('chat/', include('chat.urls')),
    path('account/', include('account.urls')),


    #Quickfix indtil der findes en profilside, der kan refereres til:
    path('accounts/profile/', views.homepage, name='homepage')
]
