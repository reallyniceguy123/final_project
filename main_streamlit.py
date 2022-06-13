import json
import time
import geojson
import requests
import geopandas as gpd
import pandas as pd
import altair as alt
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from math import ceil

import folium
import streamlit as st
from streamlit_folium import folium_static

from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

import osmnx as ox
import networkx as nx
from tg_location import get_k_nearest
from IPython.display import IFrame
import streamlit.components.v1 as components

import warnings
warnings.filterwarnings('ignore')


with st.echo(code_location='below'):
   
    st.markdown("<h1 style='text-align: center; color: black;'>Общественное питание в Москве</h1>", unsafe_allow_html=True)
    img = Image.open('rest.png')

    _, col2, _ = st.columns([1,6,1])

    with col2:
        st.image(img)

    df_geo = gpd.read_file('ao.geojson')
    df_geo.insert(df_geo.shape[1], 'centroid', df_geo.geometry.centroid)

    df = pd.read_csv('eating.csv')
    df.drop(columns = 'Unnamed: 0', axis = 1, inplace= True)
    df['OperatingCompany'] = df['OperatingCompany'].fillna("нет")

    df_ratings = pd.read_csv('eating_with_ratings.csv')
    df_ratings_gb = df_ratings[df_ratings['Rating'] != -1].groupby('AdmArea') \
       .agg({'Name':'size', 'Rating':'mean'}) \
       .rename(columns={'Name':'Количество оценок','Rating':'Средняя оценка'}) \
       .reset_index()

    df_moscow = pd.read_csv('moscow.csv')
    df_moscow.drop(columns = 'Unnamed: 0', axis = 1, inplace= True)

    st.subheader('Данные о местах общественного питания')
    if st.checkbox('Показать данные'):
        status = st.text('Загружаю...')
        count = st.slider('Кол-во строк', 1, df.shape[0], df.shape[0])
        st.write(df[:count+1])
        status.text('Загрузка данных закончена.')

    st.subheader('Визуализация')
    
    st.markdown("<h5 style='text-align: center; color: black;'>Карта Москвы с округами.</h5>", unsafe_allow_html=True)   
    m1 = folium.Map(location=[55.751999, 37.617734], zoom_start=12)

    for _, row in df_geo.iterrows():
        sim_geo = gpd.GeoSeries(row['geometry']).simplify(tolerance=0.001)
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(data=geo_j,
                           style_function=lambda x: {'fillColor': 'yellow'})
        folium.Popup(row['NAME']).add_to(geo_j)
        geo_j.add_to(m1)
    
    for _, row in df_geo.iterrows():
        lat = row['centroid'].y
        lon = row['centroid'].x
        folium.Marker(location=[lat, lon],
                  popup='Округ: {}'.format(row['NAME'])).add_to(m1)
    folium_static(m1)

    st.markdown("<h7 style='text-align: center; color: black;'>Сводная таблица со средними оценками заведений по округам.</h7>", unsafe_allow_html=True)   
    st.markdown('Было спарсено около 700 оценок заведений (больше гугл не дал).')
    st.write(df_ratings_gb)

    agree = st.checkbox('Построить график')

    if agree:
        fig_rate = go.Figure()
        fig_rate = px.scatter(df_ratings_gb, x="AdmArea", y="Средняя оценка",
                        size='Количество оценок')
        fig_rate.update_layout(barmode='stack', title="График средних оценок от округа в зависимости <br> от количества оценок.")
        st.plotly_chart(fig_rate, use_container_width=True)


    st.markdown("<h5 style='text-align: center; color: black;'>Карта Москвы с местами общ. питания.</h5>", unsafe_allow_html=True)
    count_for_show = st.slider('Количество данных на отображение', 1, ceil(df.shape[0] / 2), 500)
    eat_places = []
    i = 0
    df_temp = df.head(count_for_show)
    while i < len(df_temp.index):
        eat_places.append({'index': i, 'Coordinates': [df_temp['Latitude'][i], df_temp['Longitude'][i]], 'Location': df_temp['Address'][i], 'SeatsCount': df_temp['SeatsCount'][i], 'Name': df_temp['Name'][i]})
        i += 1
    
    marker_coordinates = [eat_pl['Coordinates'] for eat_pl in eat_places]
    marker_coordinates = [[float(x) for x in y] for y in marker_coordinates]

    m = folium.Map(location=[55.751999, 37.617734], zoom_start=12)

    info_box_template = """
        <dl>
        <dt>Название:</dt><dd>{Name}</dd>
        <dt>Адрес:</dt><dd>{Location}</dd>
        <dt>Количество посадочных мест:</dt><dd>{SeatsCount}</dd>
        </dl>
    """
    locations_info = [info_box_template.format(**point) for point in eat_places]

    for i in range(0, len(df_temp)):
        folium.Marker(
            location=[df_temp.iloc[i]['Latitude'], df_temp.iloc[i]['Longitude']],
            popup=locations_info[i],
            tooltip=df_temp.iloc[i]['Name']
        ).add_to(m)
    folium_static(m)

    st.markdown("<h5 style='text-align: center; color: black;'>Спарсенная таблица с википедии об округах Москвы.</h5>", unsafe_allow_html=True)
    st.write(df_moscow)
    st.write("Расчитать отношение числа заведений к площади округов Москвы?")
    if st.button('Посчитать'):
        selected_columns = df_moscow[["Okrug", "Площадь км²1.07.2012[4][5]"]]
        short_df = selected_columns.copy()
        short_df['Площадь км²1.07.2012[4][5]'] = short_df['Площадь км²1.07.2012[4][5]'].str.replace(',', '.')
        short_df['Отношение'] = df.groupby(['AdmArea'])['AdmArea'].count().tolist() / short_df['Площадь км²1.07.2012[4][5]'].astype(float)
        st.write(short_df)
        fig_okr = go.Figure()
        fig_okr = px.bar(short_df, x='Okrug', y='Отношение')
        fig_okr.update_layout(barmode='stack', title="График отношения кол-ва заведений <br> к площади соответствующего округа.")
        st.plotly_chart(fig_okr, use_container_width=True)


    st.markdown("<h5 style='text-align: center; color: black;'>Карта Москвы с гистограммами заведений по админ. р-нам.</h5>", unsafe_allow_html=True)
    st.write("Карта Москвы с заведениями общественного питания. Чем выше столбец, тем больше кол-во мест на единицу площади.")
    adm_area = df['AdmArea'].unique().tolist()
    opt = st.selectbox("Выберите административный район", adm_area, index=0)

    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
            latitude=55.751999,
            longitude=37.617734,
            zoom=10,
            pitch=50
        ),
        layers=[
            pdk.Layer(
                'HexagonLayer',
                data=df[df['AdmArea'] == opt],
                get_position='[Longitude, Latitude]',
                radius=100,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
                auto_highlight=True,
            )
        ],
        tooltip={"text": "{position}\nCount: {elevationValue}"}
    ))

    st.markdown("<h5 style='text-align: center; color: black;'>Количественный график мест общественного питания в зависимости от округа.</h5>", unsafe_allow_html=True)
    radio1 = st.radio(
        "Выберите библиотеку для построения графика",
        ('plotly', 'altair'))

    if radio1 == 'plotly':
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=df[df['IsNetObject'] == 'нет'].groupby('AdmArea').count().reset_index()['AdmArea'],
            x=df[df['IsNetObject'] == 'нет'].groupby('AdmArea').count().reset_index()['Name'],
            name='Не сетевое',
            orientation='h',
            marker=dict(
                color='rgba(246, 78, 139, 0.6)',
                line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
            )
        ))
        fig.add_trace(go.Bar(
            y=df[df['IsNetObject'] == 'да'].groupby('AdmArea').count().reset_index()['AdmArea'],
            x=df[df['IsNetObject'] == 'да'].groupby('AdmArea').count().reset_index()['Name'],
            name='Сетевое',
            orientation='h',
            marker=dict(
                color='rgba(58, 71, 80, 0.6)',
                line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
            )
        ))

        fig.update_layout(barmode='stack', title="Количество заведений общественного питания <br> в зависимости от округов Москвы")
        st.plotly_chart(fig, use_container_width=True)
    else:
        hist = alt.Chart(df, title="Количество заведений общественного питания в зависимости от округов Москвы").mark_bar().encode(
            x='count()',
            y='AdmArea',
            color='AdmArea'
        )
        st.altair_chart(hist, use_container_width=True)

    st.markdown("<h5 style='text-align: center; color: black;'>Статистика по среднему количеству мест в заведении <br> в зависимости от округа и принадлежности к сетевому заведению.</h5>", unsafe_allow_html=True)
    fig_4 = px.box(df, x = "SeatsCount", y="AdmArea", color='IsNetObject', title="Статистика по среднему числу мест в заведении")
    fig_4.update_traces(quartilemethod="exclusive")
    st.plotly_chart(fig_4, use_container_width=True)
    st.markdown('Если развернуть график, то можно понять, что в выборке скорее всего выбросы (из-за чего график "смещен влево").')


    st.markdown("<h5 style='text-align: center; color: black;'>Получение пяти ближайших мест общественного питания относительно вашего местоположения.</h5>", unsafe_allow_html=True)
    title_street = st.text_input('Введите вашу улицу', placeholder='Пушкинская')
    title_house = st.text_input('Введите номер дома', placeholder='7')
    knn = st.slider('Сколько ближайших мест вывести?', 1, 10, 1)
    
    if 'knn_places' not in st.session_state:
        st.session_state.knn_places = []
    if 'user_lat' not in st.session_state and 'user_lon' not in st.session_state:
        st.session_state.user_lat = 0.0
        st.session_state.user_lon = 0.0

    if st.button('Искать'):
        places_result = []
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': 'Москва, ' + title_street + ', ' + title_house,
            'format': 'json',
            'polygon': 1,
            'addressdetails':1
        }
        r = requests.get(url, params=params)
        result = r.json()[0]

        st.session_state.user_lat = float(result['lat'])
        st.session_state.user_lon = float(result['lon'])

        df_clean = df[~df['Name'].str.lower().str.contains('школа')]
        list_names = list(df_clean['Name'])
        list_addresses = list(df_clean['Address'])
        list_lat = list(df_clean['Latitude'])
        list_long = list(df_clean['Longitude'])

        places = zip(list_names, list_addresses, list_lat, list_long)
        user_location_list = [st.session_state.user_lat, st.session_state.user_lon]
        places_result = get_k_nearest(user_location_list, list(places), knn)
        st.session_state['knn_places'] = places_result
    
    select_1 = st.selectbox('Выбери место куда пойти.', options=[opt.strip() for opt in st.session_state.knn_places]) 

    mode_text = {
        'drive': 'на машине',
        'bike': 'на велосипеде',
        'walk': 'пешком'
    }

    mode_select = st.selectbox(
        'Кратчайший путь на...',
        ('пешком', 'велосипеде', 'машине'))
    
    mode_choice = {
        'машине': 'drive',
        'велосипеде': 'bike',
        'пешком': 'walk'
    }
    mode = mode_choice[mode_select]
    
    optimizer_text = {
        'времени': 'time',
        'расстоянию': 'length' 
    }

    optimizer = st.selectbox(
        'Кратчайший путь по...',
        ('времени', 'расстоянию'))


    # st.markdown('Маршрут до выбранного места.')
    if st.button('Построить маршрут') and select_1 != '':
        status_way = st.text('Загружаю кратчайший маршрут. Обычно это занимает длительное время :( Пожалуйста, подождите...')
        status_title = st.text('')

        ox.config(log_console=True, use_cache=True)

        place = 'Moscow, Russia'
        start = (st.session_state.user_lat, st.session_state.user_lon)
        end = (df[df['Name'] + '. ' + df['Address'] == select_1]['Latitude'].values[0], df[df['Name'] + '. ' + df['Address'] == select_1]['Longitude'].values[0])
        print(start)
        print(end)

        graph = ox.graph_from_place(place, network_type = mode)
        orig_node = ox.get_nearest_node(graph, start)
        dest_node = ox.get_nearest_node(graph, end)
        shortest_route = nx.shortest_path(graph, orig_node,dest_node,
                                        weight=optimizer_text[optimizer])
        shortest_route_map = ox.plot_route_folium(graph, shortest_route, 
                                                tiles='openstreetmap')

        filepath = "graph.html"
        shortest_route_map.save(filepath)
        IFrame(filepath, width=600, height=500)
        status_way.text('Маршрут загружен.')
        status_title.text('Кратчайший маршрут по {} от вашей локации до выбранной {}'.format(optimizer, mode_text[mode]))
        HtmlFile = open("graph.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        components.html(source_code)
    
    st.markdown("<h5 style='text-align: center; color: black;'>Телеграм-бот</h5>", unsafe_allow_html=True)
    st.write("Сделан телеграм бот, который по вашей геопозиции ищет ближайшие места общественного питания.")
    st.write("Код лежит в файле **tg_location.py**. Для проверки необходимо зпустить код командой python tg_location.py")
    st.write("Сам телеграмм-бот **@public_catering_bot**")