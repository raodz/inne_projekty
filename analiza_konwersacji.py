# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 12:59:16 2023

@author: Rafal
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 16:43:36 2022

@author: Rafal
"""
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 13:54:56 2022

@author: Rafal
"""
#%% Biblioteki, wczytanie plików, tworzenie listy wiadomosci
import json
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

f = open('sciezka\\do\\pliku.json')
pliki = json.load(f)
lst = pliki['messages'][::-1]       # Chronologiczna lista wiadomosci
#%% Ogólna liczba wiadomosci
print(len(lst))
#%% Opcjonalne wyrzucanie zbyt bliskich wiadomosci
# Kilka wersji if-a w zależnosci od tego, które wiadomosci chcemy przepuscic
new_list = []
for i in range(1, len(lst)):
    if (lst[i]['sender_name'] != lst[i-1]['sender_name'] or lst[i]['timestamp_ms'] - lst[i-1]['timestamp_ms'] > 300000)  and lst[i]['type']!='Call':
    #if lst[i]['type']!='Call': # and 'content' in lst[i]:
        new_list.append(lst[i])
lst = new_list
# Analizowana liczba wiadomosci
print(len(lst))
#%% Tworzenie słownika, w którym dniom przypisane są listy wiadomosci

start = 1569180391393+(94*60*1000) # Wybór dnia początkowego analizy
dni = {start: []}

for e in lst:
    if e['timestamp_ms'] < start+(1000*60*60*24):
        dni[start].append(e)
    else:
        dni[start+(1000*60*60*24)] = [e]
        start += 1000*60*60*24
#%% Tworzenie listy osób

ludzie = []
for e in pliki['participants']:
    ludzie.append(e['name'])
# Ręczne przyłączenie dodatkowych kont (jesli jakies zostały usunięte z konwersacji)
ludzie.append('konto1')
ludzie.append('konto2')
#%% Tworzenie słownika, w którym ludziom odpowiadają listy dni,w których wysłali wiadomosc (dni się mogą powtarzać)
daty = {}
for e in ludzie:
    daty[e] = []
for e in lst:
    time_in_millis = e['timestamp_ms']
    daty[e['sender_name']].append(datetime.datetime.fromtimestamp(time_in_millis / 1000.0).date())
#%% Tworzenie słownika, w którym ludziom odowiadają słowniki, w których dniom 
# odpowiadają liczby wiadomosci danej osoby w tym dniu
zliczone_daty = {}
for i in daty:
    d = {x:daty[i].count(x) for x in daty[i]}
    zliczone_daty[i] = d
#%% Maksymalne dni każdego
for e in zliczone_daty.keys():
    stats = zliczone_daty[e]
    maks = max(stats, key=stats.get)
    print(e)
    print(maks)
    print(zliczone_daty[e][maks])
#%% Suma wiadomosci w ogole dla każdego
cale_sumy = {}
for e in zliczone_daty:
    cale_sumy[e] = sum(zliczone_daty[e].values())
#%% Dodanie do słowników dla osób dni z zerem wiadomosci
zgodzi_zakres_dat = pd.date_range(start="2019-09-22",end="2022-11-16", ) # do aktualizacji jbc
zakres_dat = []
for e in zgodzi_zakres_dat:
    zakres_dat.append(e.date())
for e in zakres_dat:
    for f in zliczone_daty:
        if e not in zliczone_daty[f]:
            zliczone_daty[f][e] = 0
#%% Łączenie w jedno kont należących do tej samej osoby
zliczone_daty['Jedna_Osoba'] = {}
for e in zliczone_daty['konto1']:
    zliczone_daty['Jedna_Osoba'][e] = zliczone_daty['konto1'][e] + zliczone_daty['konto2'][e]
zliczone_daty.pop('konto1')
zliczone_daty.pop('konto2')
#%% Tworzenie ramki danych z tych słowników

df = pd.DataFrame(zliczone_daty)
df = df.T
df = df[zakres_dat]

#plt.plot(df.sum(0))
x = df.sum(0).to_dict()
print({k: v for k, v in sorted(x.items(), key=lambda item: item[1])})
#%% Opcjonalne jesli chcemy dane zsumowane (dla każdego dnia on + suma poprzednich)
'''
for i in range(df.shape[0]):
    for j in range(1, df.shape[1]):
        df.iat[i, j] += df.iat[i, j-1]
'''
#%% Zmiana nazw osób na ładniejsze

df = df.set_axis(['GJ', 'MA', 'PK', 'MK', 'RO', 'HO', 'MBU', 
                   'AT', 'AL', 'JO', 'AK', 'MBI'], axis='index')

#%% "Płynna" ramka danych uwzględniająca dla każdego jego wiadomości na przestrzni 30 ostatnich dni
plynna_df = df.copy()

for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        if j >= 30:
            x = j - 30
        else:
            x = 0
        plynna_df.iat[i, j] = sum(df.iloc[i, x:j+1])
# Ewentualne zmiany nazw dla płynnej
'''
nowe_nazwy = []
for e in plynna_df.columns:
    dt = datetime.datetime.combine(e, datetime.datetime.min.time()).timestamp()
    print(dt)
    if dt < 1569099600 + 60*60*24*30:
        nowe_nazwy.append("2019-09-22 - "+str(e))
    else:
        dd = dt - 60*60*24*30
        dd = datetime.datetime.fromtimestamp(dd).date()
        nowe_nazwy.append(str(dd)+" - "+str(e))
plynna_df.columns = nowe_nazwy
'''
#%% Opcjonalne drukowanie ramki danych do Excela

# writing to Excel
datatoexcel = pd.ExcelWriter('analiza_konwersacji.xlsx')
  
# write DataFrame to excel
plynna_df.to_excel(datatoexcel)
  
# save the excel
datatoexcel.save()

#%% Obliczanie frekwencji dla miesięcy

s = 9
msc = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
all_msc = []
for i in range(df.shape[1]):
    if df.columns[i].month == s:
        msc += np.array(df[df.columns[i]])
    else:
        if s < 12:
            s += 1
        else: s = 1
        all_msc.append(msc)
        msc = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        msc += np.array(df[df.columns[i]])
all_msc.append(msc)

#%% Rysuje wykresy dla aktywnosci kazdej osoby
# Niektóre opcje są wyłaczone - ich wlaczenie dodaje linię regresji, ale rodzi problem z 
# podpisami na osi x
sms = []
for e in all_msc:
    sms.append(sum(e))
osoby = ['GJ', 'MA', 'PK', 'MK', 'RO', 'HO', 'MBU', 
                   'AT', 'AL', 'JO', 'AK', 'MBI']

for i in range(len(osoby)):
# Aktywnosc bezwzględna
    osoba = []
    for e in all_msc:
        osoba.append(e[i])

    lista_miesiecy = pd.date_range('2019-08-10','2022-11-16', 
                  freq='MS').strftime("%Y-%b").tolist()

    plt.bar(lista_miesiecy, osoba)
    plt.xticks(rotation=90)
    plt.ylim(0, max(osoba)+10) # Skala się dostosowuje, ale można to zmienić
    plt.title(osoby[i])
    
# Aktywnosc względna

    osoba = []
    for e in all_msc:
        osoba.append(e[i])

    osoba = np.array(osoba) / np.array(sms)
    lista_miesiecy = pd.date_range('2019-08-10','2022-11-16', 
                  freq='MS').strftime("%Y-%b").tolist()
    
    plt.plot(lista_miesiecy, osoba)
    plt.xticks(rotation=90)
    plt.ylim(0, max(osoba)+.1)
    plt.title(osoby[i])
    plt.savefig('akt_wz'+osoby[i]+'.png', dpi=200)
    plt.show()
#%% Wykres wszystkich miesięcy
graph = sns.barplot(lista_miesiecy, sms)
plt.xticks(rotation=90)
plt.savefig('msc.png', dpi=200)
plt.show()

#%% Wykres różnic między miesiącami
roznice_msc = sms.copy()
for i in range(1, len(roznice_msc)):
    roznice_msc[i] = roznice_msc[i] - sms[i-1]
graph = sns.barplot(lista_miesiecy, roznice_msc)
plt.xticks(rotation=90)
plt.savefig('msc_roznice.png', dpi=200)
plt.show()

#%% Wykres sum płynnej df
plynna_df[plynna_df.columns[1:]].sum().plot.line(rot=90, color='DarkTurquoise')
plt.savefig('sumy_plynnej.png', dpi=200)
plt.show()

#%% Statystyki reakcji
reakcje = {}
for e in ludzie:
    reakcje[e] = 0
for e in lst:
    if 'reactions' in e:
        reakcje[e['sender_name']] += len(e['reactions'])
for e in reakcje:
    print(e+": "+str(reakcje[e]/cale_sumy[e]))
    
#%% Ogólna statystyka dla dni
df[df.columns[1:]].sum().plot.line(rot=90, color='DarkTurquoise')
plt.savefig('ogolna_dni.png', dpi=200)
plt.show()

#%% Ogólna statystyka dla ludzi
og_stat_lud = df.sum(axis=1)
og_stat_lud = pd.DataFrame(og_stat_lud)
og_stat_lud = og_stat_lud.set_axis(['Liczba'], axis=1)
og_stat_lud = og_stat_lud.transpose()
ax = sns.barplot(data=og_stat_lud)
ax.bar_label(ax.containers[0])
plt.xticks(rotation=45)
plt.savefig('ogolna_ludzie.png', dpi=200)
plt.show()
#%% Wykres kołowy z procentami

data = og_stat_lud.loc[:, :].values.tolist()
data = data[0]

labels = list(og_stat_lud.columns)

colors = sns.color_palette('pastel')[0:5]

plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%', 
        rotatelabels=True, textprops={'fontsize': 8})
plt.savefig('kolowy.png', dpi=200)
plt.show()