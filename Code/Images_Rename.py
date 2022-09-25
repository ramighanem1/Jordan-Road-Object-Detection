import os

def main():
	i = 0
	path="C:/Users/ramig/Desktop/New folder (2)/image/"
	for filename in os.listdir(path):
		my_dest ="image_" + str(i) + ".jpg"
		my_source =path + filename
		my_dest =path + my_dest
		os.rename(my_source, my_dest)
		i += 1
if __name__ == '__main__':
	main()

