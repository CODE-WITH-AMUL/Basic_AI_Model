from django.urls import path
from . import views

urlpatterns = [
    path('', views.chat_view, name='chat_home'),
    path('chat/', views.chat_view, name='chat_interface'),
    path('api/get-response/', views.get_ai_response, name='get_ai_response'),
    path('api/new-chat/', views.new_chat, name='new_chat'),
]