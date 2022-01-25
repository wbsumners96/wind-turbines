import inspect
from load_data import TurbineData
from model import models

import datetime
from enum import Enum
import itertools
import threading
import time
import re
import sys


class TurbinatorApp:
    def __init__(self, app_config):
        self.app_config = app_config
        self.model = None
        self.predictor_parameters = {}
        self.predictor = None
        self.data = None
        self.farm = '' 
        self.targets = []
        self.references = []
        self.times = []
        self.page = Pages.FARM

    def run(self):
        self.app_config = { 
            'title': 'TURBINATOR',
            'data_path': '~/Documents/Semester 2/TurbineProject/Data',
        }

        farms = ['ARD', 'CAU']
        
        print(self.app_config['title'])
        
        while True:
            print(f'current page is {self.page}')
            if self.page == Pages.FARM:
                self.select_farm(farms)
            elif self.page == Pages.LOAD_DATA:
                self.load_data()
            elif self.page == Pages.MODEL:
                self.select_model(models)
            elif self.page == Pages.PREDICTOR:
                self.create_predictor()
            elif self.page == Pages.TURBINES:
                self.select_turbines()
            elif self.page == Pages.RESULTS:
                self.display_results()
                self.page = Pages.EXIT
            elif self.page == Pages.EXIT:
                sys.exit()

    def select_farm(self, farms):
        while True:
            print('Choose the farm to work on: ')
            for i, farm in enumerate(farms):
                print(f'({i+1}) {farm}')
            print(f'(q) Exit application')

            user_selection = input()
            if user_selection == 'q':
                self.page = Pages.EXIT 

                return
            try:
                selected_index = int(user_selection)
                if selected_index in range(1, len(farms) + 1):
                    self.page = Pages.LOAD_DATA
                    self.farm = farms[selected_index - 1]
                    
                    return
                print('Selected index out of range.\n')
            except ValueError:
                print('Invalid selection.\n')

    def load_data(self):
        loading_complete = False 
        def show_loading_animation():
            for c in itertools.cycle(['⠟', '⠯', '⠷', '⠾', '⠽', '⠻']):
                if loading_complete:
                    break
                
                sys.stdout.write('\r' + c + ' Loading turbine data...')
                sys.stdout.flush()
                
                time.sleep(0.1)

        animation_thread = threading.Thread(target = show_loading_animation)
        animation_thread.start()

        self.data = TurbineData(self.app_config['data_path'], self.farm)

        sys.stdout.write('\rLoading complete.                        \n')
        sys.stdout.flush()

        loading_complete = True

        print(self.data.data.info())

        self.page = Pages.MODEL

    def select_model(self, models): 
        while True:
            print('Choose from the following available models: ')
            for i, model in enumerate(models):
                print(f'({i+1}) {model.__name__}')
            print('(f) Go back to farm selection')
            print('(q) Exit application')

            user_selection = input()
            if user_selection == 'q':
                self.page = Pages.EXIT

                return
            if user_selection == 'f':
                self.page = Pages.FARM

                return
            try:
                selected_index = int(user_selection)
                if selected_index in range(1, len(models) + 1):
                    selected_model = models[selected_index - 1]
                    if selected_model != self.model:
                        self.model = models[selected_index - 1]
                        self.predictor_parameters = {}
                    
                    self.page = Pages.PREDICTOR

                    return
                print('Selected index out of range.\n')
            except ValueError:
                print('Invalid selection.\n')

    def create_predictor(self):
        print(self.model.__init__.__doc__)

        print("""Enter parameters for selected model, or enter (f) to go back 
              to farm selection, (m) to go back to model selection, and (q) to 
              exit the application. Hit Return to use the currently saved 
              parameter if available (in parentheses).""")
        for parameter in inspect.signature(self.model.__init__).parameters\
                .keys():
            if parameter == 'self':
                continue 
            while True:
                current_value = f'({self.predictor_parameters[parameter]})' if \
                        parameter in self.predictor_parameters else ''
                print(f'{parameter}: {current_value} ', end='')
                user_param = input()
                if user_param == 'f':
                    self.page = Pages.FARM

                    return
                if user_param == 'm':
                    self.page = Pages.MODEL

                    return
                if user_param == 'q':
                    self.page = Pages.EXIT

                    return
                if user_param != '':
                    try:
                        self.predictor_parameters[parameter] = float(user_param)
                        break
                    except ValueError:
                        print('Invalid float value provided.')
                break

        self.predictor = self.model(**self.predictor_parameters)
        self.page = Pages.TURBINES

    def select_turbines(self):
        print('(f) Farm selection')
        print('(m) Model selection')
        print('(p) Predictor construction')
        print('(q) Exit application')
        
        targets = []
        while True:
            print('Enter target turbine IDs separated by commas: ')
            try:
                user_input = input()
                if user_input == 'f':
                    self.page = Pages.FARM
                    return
                if user_input == 'm':
                    self.page = Pages.MODEL
                    return
                if user_input == 'p':
                    self.page = Pages.PREDICTOR
                    return
                if user_input == 'q':
                    self.page = Pages.EXIT
                    return

                user_targets = user_input.replace(' ', '').split(',')
                for user_target in user_targets:
                    targets.append(int(user_target))

                self.targets = targets
                self.page = Pages.RESULTS
                
                break
            except ValueError:
                print('Invalid target.')

        references = []
        while True:
            print('Enter reference turbine IDs separated by commas: ')
            try:
                user_input = input()
                if user_input == 'f':
                    self.page = Pages.FARM
                    return
                if user_input == 'm':
                    self.page = Pages.MODEL
                    return
                if user_input == 'p':
                    self.page = Pages.PREDICTOR
                    return
                if user_input == 'q':
                    self.page = Pages.EXIT
                    return

                user_references = user_input.replace(' ', '').split(',')
                for user_reference in user_references:
                    references.append(int(user_reference))

                self.references = references
                self.page = Pages.RESULTS

                break
            except ValueError:
                print('Invalid references.')

        times = []
        while True:
            print('Enter times in ISO format YYYY-MMM-DD HH:MM:SS separated by commas: ')
            try:
                user_input = input()
                if user_input == 'f':
                    self.page = Pages.FARM
                    return
                if user_input == 'm':
                    self.page = Pages.MODEL
                    return
                if user_input == 'p':
                    self.page = Pages.PREDICTOR
                    return
                if user_input == 'q':
                    self.page = Pages.EXIT
                    return

                user_times = re.sub(r',\s*', user_input, ',').split(',')
                for user_time in user_times:
                    times.append(datetime.datetime.fromisoformat(user_time))

                self.times = times 
                self.page = Pages.RESULTS

                break
            except ValueError:
                print('Invalid times.')

    def display_results(self):
        print(self.predictor.predict(self.data, self.targets, self.references,
            self.times))


class Pages(Enum):
    FARM = 0
    LOAD_DATA = 1
    MODEL = 2
    PREDICTOR = 3
    TURBINES = 4
    RESULTS = 5
    EXIT = 6


if __name__ == '__main__':
    app = TurbinatorApp({})
    app.run()

