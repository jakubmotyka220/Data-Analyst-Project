import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('Airbnb_Open_Data.csv')

#sprawdzenie typów danych w bazie
print(data.info())
#sprawdzenie niestandardowych znaków w kolumnie price
non_numeric_chars = data['price'].astype(str).str.extractall(r'([^0-9.])')[0].unique()
print(f"\nZnalezione znaki niestandardowe: {non_numeric_chars}")
#czyszczenie kolumny price z niestandardowych znaków
data['price'] = data['price'].replace(r'[\$,\s]', '', regex=True)
data['price'] = pd.to_numeric(data['price'], errors='coerce')
#usunięcie wierszy z brakującymi wartościami
clear_data = data.dropna(subset=['price', 'neighbourhood', 'room type'])
#sprawdzenie poprawności danych
print(f'''\nDescribe dla kolumny availability 365:\n{clear_data["availability 365"].describe()} \nDescribe dla kolumny number of reviews:\n{clear_data["number of reviews"].describe()}
      \nDescribe dla kolumny minimum nights:\n{clear_data["minimum nights"].describe()} \nDesdcribe dla kolumny calculated host listings count:\n{clear_data["calculated host listings count"].describe()}''')
#czyszczenie danych z nieprawidłowych wartości w kolumnie availability 365, minimum nights
clear_data = clear_data[(clear_data['availability 365'] >= 0) & (clear_data['availability 365'] <= 365)]
clear_data = clear_data[(clear_data['minimum nights'] > 0) & (clear_data['minimum nights'] <= 365)]
print(f'''\nDescribe dla kolumny availability 365:\n{clear_data["availability 365"].describe()} 
      \nDescribe dla kolumny minimum nights:\n{clear_data["minimum nights"].describe()}''')
#poprawa dzielnic
clear_data['neighbourhood group'] = clear_data['neighbourhood group'].replace(
{
    'manhatan': 'Manhattan', 
    'brookln': 'Brooklyn'
})


#sprawdzenie korelacji między ceną a innymi zmiennymi liczbowymi
numeric_data = clear_data.select_dtypes(include=['number'])
numeric_data = numeric_data.drop(columns=['id', 'host id', 'lat', 'long'])
correlation = numeric_data.corr()
print(f"\nMacierz korelacji dla ceny:\n{correlation['price'].sort_values(ascending=False)}")
print("\nŻadna z korelacji nie jest na tyle wysoka, aby uznać ją za istotną.")
#wizualizacja macierzy korelacji
plt.figure(figsize=(10, 6))
heatmap = sns.heatmap(numeric_data.corr()[['price']].sort_values(by='price', ascending=False), 
                      annot=True, 
                      cmap='coolwarm', 
                      fmt=".4f",
                      linewidths=0.5,
                      annot_kws={"size": 10})
heatmap.set_xticks([]) # ukrycie etykiet osi x
plt.yticks(rotation=0)
plt.tight_layout()
plt.title('Korelacja zmiennych względem ceny')
plt.show()
#średnia cena w zależności od typu pokoju
room_type_price = clear_data.groupby('room type')['price'].mean().sort_values(ascending=False)
print(f"\nŚrednia cena według typu pokoju:\n{room_type_price}")

#średnia cena oraz średnia liczba dostępnych dni według dzielnicy (sortowane po cenie malejąco)
neighbourhood_price = clear_data.groupby('neighbourhood').agg(avg_price=('price', 'mean'), avg_availability_365=('availability 365', 'mean')).sort_values(by='avg_price', ascending=False)

#dodanie dodatkowych kolum by ułatwić analizę dzielnic
neighbourhood_price['global_avg_price'] = clear_data['price'].mean()
neighbourhood_price['global_median_price'] = clear_data['price'].median()
neighbourhood_price['number_of_listings'] = clear_data['neighbourhood'].value_counts()
neighbourhood_price['possibility_profit'] = (365 - neighbourhood_price['avg_availability_365']) * neighbourhood_price['avg_price']

#wyświetlenie posortowanych danych
print(f"\nDzielnice posortowane po średniej cenie:\n{neighbourhood_price}")
print(f"\nDzielnice posortowane po średniej liczbie dostępnych dni w roku:\n{neighbourhood_price.sort_values(by='avg_availability_365')}")
print(f"\nDzielnice posortowane po możliwości zysku:\n{neighbourhood_price.sort_values(by='possibility_profit', ascending=False)}")

#wyświetlenie (top 3 każdego pokoju) średniego zysku możliwego do uzyskania w każdej dzielnicy po każdym typie pokoju
top_profit_by_room_type = clear_data.groupby(['neighbourhood', 'room type']).agg(avg_price=('price', 'mean'), avg_availability_365=('availability 365', 'mean'))
top_profit_by_room_type['possibility_profit'] = (365 - top_profit_by_room_type['avg_availability_365']) * top_profit_by_room_type['avg_price']
top_profit_by_room_type = top_profit_by_room_type.reset_index()
print(f"\nTop 3 możliwości zysku według typu pokoju w każdej dzielnicy:\n{top_profit_by_room_type.sort_values(by='possibility_profit', ascending=False).groupby('room type').head(3)}")

#przedstawienie wniosków
print("\nWnioski:")
print("\nW zależności od typu pokoju, możemy wyróżnić różne propozycje w jakiej dzielnicy warto wynająć lokal:")
for room_type in top_profit_by_room_type['room type'].unique():
    top_neighbourhoods = top_profit_by_room_type[top_profit_by_room_type['room type'] == room_type].sort_values(by='possibility_profit', ascending=False).head(3)
    print(f"\nDla typu pokoju '{room_type}':")
    for index, row in top_neighbourhoods.iterrows():
        print(f"- Dzielnica: {row['neighbourhood']}, Potencjalny przychód: {row['possibility_profit']:.2f}")
#Export danych
clear_data.reset_index().to_csv('tableau_clear_data.csv', index=False)
print(f"\nPlik CSV został wygenerowany pomyślnie.")