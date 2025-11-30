

if __name__ == "main__":

    car_moved = True

    while True:
        #car_moved = state_received()
        if(car_moved):
            print("car moved")
        
        elif(car_moved == False):
            print("car stopped")
        