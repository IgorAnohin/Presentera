from sys import argv

def calc(text):
	try:
	    print("Hello " + text)
	except Exception as e:
		print(e)

if __name__ == "__main__":
    print(calc(argv[1]))
