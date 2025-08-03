import dragon
import multiprocessing as mp


def main():
    mp.set_start_method("dragon")
    print("Hello World!")


if __name__ == "__main__":
    main()
