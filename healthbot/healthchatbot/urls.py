from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings
from healthchatbot import views

urlpatterns = [
    path("chat-link", views.user_conversation)
]