BASE_URL="http://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/"

all: chn_provinces.shp

chn_provinces.shp: ne_10m_admin_1_states_provinces.shp
	ogr2ogr \
	  -lco ENCODING=UTF-8 \
	  -where "adm0_a3 ='CHN'" \
	  $@ $<

%.shp: %.zip
	unzip -j $<

%.zip:
	curl -L -O $(BASE_URL)$@
