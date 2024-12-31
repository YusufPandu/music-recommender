from django.shortcuts import render, redirect
from .models import Song, Genre, Artist, Playlist
import os
import pandas as pd
from django.conf import settings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import librosa
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import subprocess


def home(request):
    return render(request, 'recommender/home.html')

def playlist(request):
    return render(request, 'recommender/playlist.html')

def music_recommender(request):
    # Define dataset path
    dataset_path = os.path.join(settings.BASE_DIR, 'dataset', 'features_30_sec.csv')
    data = pd.read_csv(dataset_path)

    # Process the data
    labels = data[['label']]
    data = data.drop(columns=['length', 'label'])
    data_scaled = MinMaxScaler().fit_transform(data)

    # Cosine similarity
    similarity = cosine_similarity(data_scaled)
    sim_df = pd.DataFrame(similarity, index=labels.index, columns=labels.index)

    # Recommendation logic
    now_playing = 'classical.00003.wav'
    recommended_songs = sim_df[now_playing].sort_values(ascending=False).head(5)

    return render(request, 'recommender/recommendations.html', {
        'now_playing': now_playing,
        'recommendations': recommended_songs
    })

def recommendations(request):
    # Dummy data for now
    songs = [
        {"title": "Song 1", "artist": {"name": "Artist 1"}, "genre": {"name": "Genre 1"}},
        {"title": "Song 2", "artist": {"name": "Artist 2"}, "genre": {"name": "Genre 2"}},
    ]
    return render(request, 'recommender/recommendations.html', {'songs': songs})

import mimetypes
from django.core.exceptions import ValidationError


def upload_music(request):
    if request.method == 'POST' and request.FILES['music_file']:
        # Ambil file yang diunggah
        music_file = request.FILES['music_file']

        # Validasi file
        try:
            validate_audio_file(music_file)
        except ValidationError as e:
            # Tampilkan pesan error jika file tidak valid
            return render(request, 'recommender/upload.html', {
                'error_message': str(e)
            })

        # Simpan file
        fs = FileSystemStorage()
        file_path = fs.save(music_file.name, music_file)
        file_url = fs.url(file_path)

        # Ekstrak fitur dari file yang diunggah
        full_path = os.path.join(settings.MEDIA_ROOT, file_path)
        features = extract_features(full_path)

        # Muat dataset dan rekomendasikan lagu
        dataset_path = os.path.join(settings.BASE_DIR, 'dataset', 'features_30_sec.csv')
        recommendations = recommend_music(features, dataset_path)
        recommendations = recommendations[::-1]
        # Pass rekomendasi lagu ke template
        return render(request, 'recommender/recommendations.html', {
            'file_url': file_url,
            'recommendations': recommendations,
        })

    return render(request, 'recommender/upload.html')

def validate_audio_file(file):
    # Ekstensi file yang diperbolehkan
    allowed_extensions = ['.mp3', '.wav', '.m4a']
    # Tipe MIME yang diperbolehkan
    allowed_mime_types = ['audio/mpeg', 'audio/wav', 'audio/x-m4a']

    # Periksa ekstensi file
    ext = os.path.splitext(file.name)[1].lower()
    if ext not in allowed_extensions:
        raise ValidationError("File yang diunggah harus berformat MP3, WAV, atau M4A.")

    # Periksa MIME type
    mime_type, _ = mimetypes.guess_type(file.name)
    if mime_type not in allowed_mime_types:
        raise ValidationError("File yang diunggah harus berupa file audio yang valid.")


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    features = {
        'chroma_stft': np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
        'rmse': np.mean(librosa.feature.rms(y=y)),
        'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        'rolloff': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(y)),
    }
    return pd.DataFrame([features])

def recommend_music(features, dataset_path):
    # Load dataset
    data = pd.read_csv(dataset_path)

    # Separate filenames, labels, and numeric features
    filenames = data['filename']
    labels = data['label']
    numeric_data = data.drop(columns=['filename', 'label', 'length'], errors='ignore')

    # Align features to match dataset columns
    features = features.reindex(columns=numeric_data.columns, fill_value=0)

    # Scale numeric data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(numeric_data)

    # Scale uploaded file features
    features_scaled = scaler.transform(features)

    # Compute similarity
    similarity = cosine_similarity(features_scaled, data_scaled)
    top_indices = np.argsort(similarity[0])[-5:][::-1]

    # Retrieve recommended filenames and labels
    recommended_songs = filenames.iloc[top_indices].values.tolist()
    recommended_labels = labels.iloc[top_indices].values.tolist()

    # Create a list of recommendations

    recommendations = [
    {
        "filename": f"dataset/genres_original/{label}/{song}",
        "label": label,
        "songname": song,
    }
    for song, label in zip(recommended_songs, recommended_labels)
]

    return recommendations




def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # Compute features
    features = {
        'chroma_stft_mean': np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
        'chroma_stft_var': np.var(librosa.feature.chroma_stft(y=y, sr=sr)),
        'rmse_mean': np.mean(librosa.feature.rms(y=y)),
        'rmse_var': np.var(librosa.feature.rms(y=y)),
        'spectral_centroid_mean': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'spectral_centroid_var': np.var(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'spectral_bandwidth_mean': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        'spectral_bandwidth_var': np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        'rolloff_mean': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        'rolloff_var': np.var(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        'zero_crossing_rate_mean': np.mean(librosa.feature.zero_crossing_rate(y)),
        'zero_crossing_rate_var': np.var(librosa.feature.zero_crossing_rate(y)),
    }

    # Add more features if needed to match dataset
    return pd.DataFrame([features])
