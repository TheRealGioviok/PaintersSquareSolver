###############################################################################################
## DISCLAIMER:                                                                               ##
## This is the original code for the first python version of the game,                       ##
## It is conceded from its author (Bruno Montalto) to be used for educational purposes only. ##
## Any other use is forbidden.                                                               ##
## The author is not responsible for any damage caused by the use of this code.              ##
###############################################################################################

import pygame
import random as rn
import math,time
import colorsys
from copy import *
import sys
sys.setrecursionlimit(2000)
pygame.init()
pygame.font.init()
OFFSET = 29
w,h = size = 480,480+OFFSET
wn = pygame.display.set_mode(size)
pygame.display.set_caption("Painter's Square")
font1 = pygame.font.SysFont("Arial",14)

class TextButton:
    def __init__(self,x,y,size,color,text):
        self.x = x
        self.y = y
        self.size = size
        self.text = text
        self.color = color
        self.img = font1.render(text,True,(255,255,255))
        self.selected = False

    def setText(self,text):
        self.text = text
        self.img = font1.render(text,True,(255,255,255))

    def draw(self,surf,down):
        pygame.draw.rect(surf,(self.color[0]-20,self.color[1]-20,self.color[2]-20) if down else self.color,(self.x - self.size[0]/2,self.y-self.size[1]/2,self.size[0],self.size[1]))
        size = self.img.get_size()
        surf.blit(self.img,(self.x - size[0]/2,self.y-size[1]/2))

    def inTouch(self,mx,my):
        return mx > self.x - self.size[0]/2 and mx < self.x + self.size[0]/2 and my > self.y - self.size[1]/2 and my < self.y + self.size[1]/2

def rgb(h,s,v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h/100,s/100,v/100))

class Puzzle:
    def __init__(self, size = 3, colors = 6):
        self.n = size
        self.colors = colors
        self.brush = [[1,0,1],
                      [0,1,0],
                      [1,0,1]]
        self.new(False)

    def new(self, shuffle = True):
        self.grid = []
        c=0
        for i in range(self.n):
            temp = []
            for i in range(self.n):
                temp.append(c)
            self.grid.append(temp)
        if shuffle: self.shuffle(self.n*10)

    def play(self,row,col):
        for i in range(len(self.brush)):
            for j in range(len(self.brush[0])):
                tX, tY = row - len(self.brush[0])//2 + j, col  - len(self.brush)//2 + i
                if 0 <= tX < self.n and 0 <= tY < self.n and self.brush[i][j]:
                    self.grid[tY][tX] = (self.grid[tY][tX]+1)%self.colors
        return self.is_solved()

    def undo(self,row,col):
        for i in range(len(self.brush)):
            for j in range(len(self.brush[0])):
                tX, tY = row - len(self.brush[0])//2 + j, col  - len(self.brush)//2 + i
                if 0 <= tX < self.n and 0 <= tY < self.n and self.brush[i][j]:
                    self.grid[tY][tX] = (self.grid[tY][tX]-1)%self.colors
        return self.is_solved()

    def shuffle(self,times = 100):
        for i in range(times):
            row,col = rn.randint(0,self.n-1),rn.randint(0,self.n-1)
            self.play(row,col)




    def is_solved(self):
        first = self.grid[0][0]
        for i in range(self.n):
            for j in range(self.n):
                if self.grid[i][j] != first: return False
        return True


    def oneToSolve(self):
        clone = deepcopy(self)
        for i in range(self.n):
            for j in range(self.n):
               if clone.play(j,i): return (j,i) # push
               for k in range(self.colors-1):# pop
                    clone.play(j,i)
        return False

    def eval2(self,moves={},depth=0):
        ots = self.oneToSolve()
        if ots:
            print("trovata!",ots)
            return (ots,1)
        for i in range(self.n):
            for j in range(self.n):
                self.play(j,i) # push
                
                if not (j,i) in moves: moves[(j,i)] = 0
                if moves[(j,i)] == self.colors:
                    self.undo(j,i)# pop
                    moves[(j,i)]=0                
                    return (None,0)
                
                moves[(j,i)] += 1
                eval = self.eval2(moves,depth+1)
                if eval[1]:
                    self.undo(j,i)
                    #moves[(j,i)]-=1
                    return((j,i),eval[1])
                    

                self.undo(j,i)# pop
                #moves[(j,i)]-=1
        return (None,0)

    def updateDraw(self,time=60):
        return
        self.draw(wn)
        pygame.display.update()
        clock.tick(time)


    def eval(self,move,moves={},depth=0):
        m = self.play(*move)
        self.updateDraw()

        if depth == 0: self.undo(*move)
        if m: #mossa vincente
            self.updateDraw(0.5)
            print("found")
            return (True,depth)
        


                
        if move in moves: #mossa ripetuta n colori volte
            if moves[move] == self.colors:
                moves[move] = 0
                return (False,depth)
        
        for i in range(self.n):
            for j in range(self.n):
                if not (i,j) in moves: moves[(i,j)] = 0
                moves[(i,j)]+=1      
                v = self.eval((i,j),moves,depth+1)
                self.undo(i,j)
                self.updateDraw()
                if v[0]: return v
                
        return (False,depth)

    def test(self):
        best = (None,1000000)
        for i in range(self.n):
            for j in range(self.n):
                v = self.eval((i,j))
                print(i*3+j,(i,j),v)
                if v[1] == 0:
                    print("|-------- best:",((i,j),v[1]),"---------|")
                    return
                if v[1]< best[1]: best = ((i,j),v[1])
        print("|-------- best:",best,"---------|")
        #self.play(*best[0])
        

    def playBest(self):
        eval = self.eval()
        print("val:",eval)
        if eval[0]:
            self.play(*eval[0])
            return True
        return False






    def draw(self,wn,tX="",tY=""):
        tile_size = w//self.n
        for i in range(self.n):
            for j in range(self.n):
                color = rgb((self.grid[i][j]* 100/self.colors)%101,90 if (i!=tY or j!=tX) else 50,100)
                pygame.draw.rect(wn, color, (j * tile_size + 1,OFFSET + i * tile_size + 1, tile_size - 2, tile_size - 2))
    
    def icon(self,surf):
        tile_size = 16//self.n
        for i in range(self.n):
            for j in range(self.n):
                color = rgb((self.grid[i][j]* 100/self.colors)%101,90,100)
                pygame.draw.rect(surf, color, (j * tile_size , i * tile_size , tile_size , tile_size ))
    
                

clock = None
def main():
    global clock
    clock = pygame.time.Clock()
    game = Puzzle()
    mouseD = False
    buttons = [TextButton(w/6,15,(75,20),(50,50,50),"New Game"),TextButton((w*2)/3 + w/6,15,(70,20),(50,50,50),"Size: 3")]
    buttonDown = None
    temp_size = 3

    timer = 0.0
    timer_on = False
    timer_display = TextButton(w/2,15,(75,20),(50,50,50),str(timer)[:5])

    #Icon
    icon_game = Puzzle(3)
    icon_game.shuffle()
    icon = pygame.Surface((16,16))
    #Icon Draw
    icon.fill((255,255,255))
    icon_game.icon(icon)
    pygame.display.set_icon(icon)
    #Easter egg :D
    if icon_game.is_solved():
        game.n = 60
        game.new()
        timer_on = True
        temp_size = 2
        buttons[1].setText("Size: 60?!")
    new_game_blink = not icon_game.is_solved()
    del icon_game #non so se da problemi
    while 1:
        mx,my = mousePos = pygame.mouse.get_pos()
        tileX,tileY = mx//(w//game.n),(my-OFFSET)//(w//game.n)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                anyTouch = False
                for b in buttons:
                    if b.inTouch(mx,my):
                        buttonDown = b
                        anyTouch = True
                        break                
                if not anyTouch and timer_on: mouseD = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if buttonDown:
                    if buttonDown.inTouch(mx,my):
                        if buttonDown.text == "New Game":
                            if new_game_blink:
                                new_game_blink = False
                                buttonDown.color = (50,50,50)
                            game.n = temp_size
                            game.new()
                            timer = 0
                            timer_on = True
                        elif buttonDown.text.split(": ")[0] == "Size":
                            temp_size = (temp_size+1)*int(temp_size<6) + 3*int(temp_size>=6)
                            buttonDown.setText("Size: "+str(temp_size))
                elif my >= OFFSET :
                    if event.button == 1:
                        if game.play(tileX,tileY):
                            timer_on = False
                    elif event.button == 3:
                        if game.undo(tileX,tileY):
                            timer_on = False
                    mouseD = False
                buttonDown = None
            elif event.type == pygame.KEYUP:
                break
                game.test()
                break
                if event.key == pygame.K_SPACE:
                    game.playBest()
                
        #######################
        timer += 1/60 * int(timer_on)
        timer_display.setText(str(timer)[:5])

        if new_game_blink:
            color = 50 + 15*math.sin(time.time()*6)+15
            buttons[0].color = (color,color,color)
        
        #Drawing 
        wn.fill((20,20,20))
        #pygame.draw.rect(wn,(255,235,140),(tileX*(w//game.n),tileY*(w//game.n),(w//game.n),(w//game.n)))
        if mouseD:
            game.draw(wn,tileX,tileY)
        else:
            game.draw(wn)
        for b in buttons:
            b.draw(wn,buttonDown is b)
        timer_display.draw(wn,timer_on)
        #######################
        pygame.display.update()
        clock.tick(60)
        
main()
quit()