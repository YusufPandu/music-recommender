from django.contrib import admin
from .models import Genre, Artist, Song, Playlist

admin.site.register(Genre)
admin.site.register(Artist)
admin.site.register(Song)
admin.site.register(Playlist)
