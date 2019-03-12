# Info: Um die Daten als csv-File ausgeben zu lassen, muessen wir bloss ueber das Terminal aus unser Projektfolder zugreifen und folgenden Befehl starten
# scrapy crawl wettercsv -o wetter.csv

# Import von Scrapy
import scrapy
# Um die Daten als .csv auszugeben oder um sie in einer Datenbank zu speichern muessen wir das ausgegebene Dictionary in einen Container Item umwandeln. Das geschieht mit Hilfe der items.py
# Daher importieren wir die Klasse MeinscrapyItem aus dem File items.py
from ..items import MeinscrapyItem

# Erstellen einer Klasse, die aus Scrapy vereerbt wird
class WetterSpider (scrapy.Spider):
    # Name fuer die Spider vergeben (die Variable muss zwingend name heissen). Diesen Namen rufen wir ueber die Command Prompt auf. Bspw. scrapy crawl wettertemperaturcsv
    name = "wettertemperaturcsv"
    # Liste mit den Webseiten, die wir Crawlen wollen (muss zwingend start_url heissen)
    start_urls = [
        'https://www.meteoblue.com/de/wetter/vorhersage/woche/st.-gallen_schweiz_2658822'
    ]

    # Erstellen der Methode parse (Varibale response wird den Sourcecode der Internetseite erhalten, die wir Crawlen wollen)
    def parse (self, response):
        # Erstellen einer neuen Variable (Instanz) der Item-Klasse um die Daten als csv ausgeben zu lassen
        items = MeinscrapyItem()

        # Erstellen einer Variable fuer die Temperatur. Response gibt eine Liste zurueck
        aktuelle_temperaturen = response.css(".tab_temp_max::text").extract(),
        # Ausgabe mit xpath Selektor
        minimale_temperatur = response.xpath("//div[@class='tab_temp_max']/text()").extract()

        # Uebergabe des Outputs
        items ['aktuelle_temperaturen'] = aktuelle_temperaturen
        items ['minimale_temperatur'] = minimale_temperatur

        # Es reicht jetzt nur noch die Items auszugeben
        yield items
