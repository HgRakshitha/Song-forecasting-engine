import streamlit as st
import pickle
import pandas as pd
import requests

def fetch_poster(music_title):
    url = f"https://jiosaavn-api.vercel.app/search?query={music_title}"
    response = requests.get(url)

    if response.status_code != 200:
        st.error("Error fetching data from API")
        return None

    data = response.json()

    # Inspecting the response structure
    if 'results' in data and len(data['results']) > 0:
        try:
            return data['results'][0]['image']
        except (KeyError, IndexError) as e:
            st.error(f"Error fetching poster: {e}")
            return None
    else:
        st.error("No results found in the API response")
        return None

def recommend(musics):
    music_index = music[music['Song-Name'] == musics].index[0]
    distances = similarity[music_index]
    music_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_music = []
    recommended_music_poster = []
    for i in music_list:
        music_title = music.iloc[i[0]]['Song-Name']
        recommended_music.append(music_title)
        poster = fetch_poster(music_title)
        if poster:
            recommended_music_poster.append(poster)
        else:
            # Use a placeholder image if poster not found
            recommended_music_poster.append('https://via.placeholder.com/150?text=No+Image')
    return recommended_music, recommended_music_poster

# Load data
music_dict = pickle.load(open('C:/Users/ADMIN/musicrec.pkl', 'rb'))
music = pd.DataFrame(music_dict)

similarity = pickle.load(open('C:/Users/ADMIN/similarities.pkl', 'rb'))

st.title('Music Recommendation System')

selected_music_name = st.selectbox('Select a music you like', music['Song-Name'].values)

if st.button('Recommend'):
    names, posters = recommend(selected_music_name)

    col1, col2, col3, col4, col5 = st.columns(5)
    if len(names) > 0:
        with col1:
            st.text(names[0])
            st.image(posters[0], use_column_width=True)
        with col2:
            st.text(names[1])
            st.image(posters[1], use_column_width=True)
        with col3:
            st.text(names[2])
            st.image(posters[2], use_column_width=True)
        with col4:
            st.text(names[3])
            st.image(posters[3], use_column_width=True)
        with col5:
            st.text(names[4])
            st.image(posters[4], use_column_width=True)
    else:
        st.error("No recommendations found")
