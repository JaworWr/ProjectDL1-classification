DATA_URL = http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz

download:
	wget -nv --show-progress -O - $(DATA_URL) | tar -xzC ./data