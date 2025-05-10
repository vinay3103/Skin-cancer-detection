from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('home/', views.home, name="home"),
    path('', auth_views.LoginView.as_view(template_name='main/login.html'), name="login"),
    path('signup/', views.signup, name="signup"),
    path('login/', auth_views.LoginView.as_view(template_name='main/login.html'), name="login"),
    path('logout/', auth_views.LogoutView.as_view(), name="logout"),
    path('predict/', views.predict_disease, name="predict"),
    path('diseases/', views.diseases_info, name="diseases_info"),
]
