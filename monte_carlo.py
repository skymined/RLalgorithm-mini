import random 

# grid(환경) 클래스 설정

class GridWorld():
    def __init__(self):
        self.x=0
        self.y=0
    def stop(self, a):
        if a==0:
            self.move_right()
        elif a==1:
            self.move_left()
        elif a==2:
            self.move_up()
        elif a==3:
            self.move_down()

        reward = -1
        done = self.is_done()
        return (self.x, self.y), reward, done
    def move_right(self):
        self.y +=1