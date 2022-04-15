from marlgrid.base import MultiGridEnv, MultiGrid
from marlgrid.objects import *
import random

class ChoosePathGrid(MultiGridEnv):
    mission = "get to the green square"
    metadata = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.reset()


    def _gen_grid(self, width=11, height=11):
        self.grid = MultiGrid((width, height))
        self.grid.wall_rect(0, 0, width, height)

        # とりあえず Wallで埋める．
        for x in range(1, width-1):
            for y in range(1, height-1):
                self.put_obj(Wall(), x, y)
        # １つの分岐点を含む路を生成する．
        for y in range(1, height-1):
            self.put_obj(None, 1, y)

        path1_y = random.randint(2, width -2) 
        

        while True: # TODO: 無限ループを外す
            path2_y = random.randint(2, width -2) 
            diff = path2_y - path1_y
            if diff <= -2 or diff >= 2:
                break
        
        for x in range(1, self.width-1):
            self.put_obj(None, x, path1_y)
            self.put_obj(None, x, path2_y)

        # 2つの分かれ道のどちらかの奥にゴールを配置する
        self.put_obj(Goal(color="green", reward=1), width-2, path1_y)

        self.agent_spawn_kwargs = {'top':(1,1), 'size':(1,1)}
        self.place_agents(**self.agent_spawn_kwargs)
