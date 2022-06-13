import logging
import pandas as pd
from heapq import nsmallest
from math import asin, cos, radians, sin, sqrt

from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

NETOBJECT, LOCATION = range(2)
param = ""

def start(update, _):
    update.message.reply_text(
        'Подсказываю места, где покушать. '
        'Команда /cancel, чтобы прекратить разговор.')
    
    reply_keyboard = [['сетевое', 'нет', 'любое']]
    markup_key = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
    update.message.reply_text(
        'Вас интересует сетевое заведение (например, Додо-пицца) или нет. '
        'Или в целом любое?',
        reply_markup=markup_key)
    return NETOBJECT

def netobject(update, _):
    global param
    param = update.message.text
    update.message.reply_text('Отправь геолокацию для получения списка мест общепитов.')
    return LOCATION
        


### FROM: https://stackoverflow.com/questions/59736682/find-nearest-location-coordinates-in-land-using-python
def dist_between_two_lat_lon(name, *args):
    lat1, lat2, long1, long2 = map(radians, args)
    dist_lats = abs(lat2 - lat1) 
    dist_longs = abs(long2 - long1) 
    a = sin(dist_lats/2)**2 + cos(lat1) * cos(lat2) * sin(dist_longs/2)**2
    c = asin(sqrt(a)) * 2
    radius_earth = 6378
    return { name : c * radius_earth }
### END FROM

def get_k_nearest(location, data, k):
    result = {}
    distance = [dist_between_two_lat_lon(p[0] + ". " + p[1], location[0], p[2], location[1], p[3]) for p in data]
    for i in distance:
        result.update(i)
    #return min(result, key=result.get)
    return nsmallest(k, result, key = result.get)

def location(update, _):
    data = pd.read_csv('./eating.csv')
    data = data[~data['Name'].str.lower().str.contains('школа')]
    user = update.message.from_user
    user_location = update.message.location
    logger.info(
        "Местоположение %s: %f / %f", user.first_name, user_location.latitude, user_location.longitude)
    logger.info(
        "IsNetObject %s", param)
        
    if param == "сетевое":
        list_names = list(data[data['IsNetObject'] == 'да']['Name'])
        list_addresses = list(data[data['IsNetObject'] == 'да']['Address'])
        list_lat = list(data[data['IsNetObject'] == 'да']['Latitude'])
        list_long = list(data[data['IsNetObject'] == 'да']['Longitude'])
    elif param == "нет": 
        list_names = list(data[data['IsNetObject'] == 'нет']['Name'])
        list_addresses = list(data[data['IsNetObject'] == 'нет']['Address'])
        list_lat = list(data[data['IsNetObject'] == 'нет']['Latitude'])
        list_long = list(data[data['IsNetObject'] == 'нет']['Longitude'])
    else:
        list_names = list(data['Name'])
        list_addresses = list(data['Address'])
        list_lat = list(data['Latitude'])
        list_long = list(data['Longitude'])

    places = zip(list_names, list_addresses, list_lat, list_long)
    user_location_list = [user_location.latitude, user_location.longitude]
    places_result = get_k_nearest(user_location_list, list(places), 4)

    temp_answer = ['· ' + place + '\n' for place in places_result]
    answer = ('').join(temp_answer)
    answer = "Ближайшие места общепита:\n{}".format(answer)

    update.message.reply_text(answer, reply_markup=ReplyKeyboardRemove())

    return ConversationHandler.END

def skip_location(update, _):
    user = update.message.from_user
    logger.info("Пользователь %s не отправил локацию.", user.first_name)
    update.message.reply_text(
        'Без геолокации не узнаешь ближайшие места :('
    )
    return ConversationHandler.END

def cancel(update, _):
    user = update.message.from_user
    logger.info("Пользователь %s отменил разговор.", user.first_name)
    update.message.reply_text(
        'Захочешь получить места - пиши!', 
        reply_markup=ReplyKeyboardRemove()
    )
    return ConversationHandler.END


if __name__ == '__main__':
    updater = Updater("5587061316:AAHv01yXEw8B1p5PlJNYtFXJQBZAhDsiVLY")
    dispatcher = updater.dispatcher

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            NETOBJECT: [MessageHandler(Filters.regex('^(сетевое|нет|любое)$'), netobject)],
            LOCATION: [
                MessageHandler(Filters.location, location),
                CommandHandler('skip', skip_location),
            ]
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )
    dispatcher.add_handler(conv_handler)
    updater.start_polling()
    updater.idle()