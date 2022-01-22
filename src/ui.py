from load_data import TurbineData

import itertools
import threading
import time
import sys


class TurbinatorApp:
    def run(self):
        # Load application configuration from config.toml.
        APP_CONFIG = { 
            'title': 'TURBINATOR',
            'data_path': '~/Documents/Semester 2/TurbineProject/Data'
        }
        
        # Print application title.
        print(APP_CONFIG['title'])

        # Ask user for farm choice and load farm data.
        farm = self.select_farm(['ARD', 'CAU'])
        if farm is None:
            sys.exit()    
        
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

        data = TurbineData(APP_CONFIG['data_path'], farm)
        sys.stdout.write('\rLoading complete.                        \n')
        loading_complete = True

        print(data.data.info())

        # Load available models from appconfig and have user select one. 
        # Have user enter model parameters and construct predictor.
        # Ask user for targets, references, and times.
        # Run predictor and display results.

    def select_farm(self, farms):
        """
        Display farm selection screen.

        Parameters
        ----------
        farms: list[str]
            Available farms to choice.

        Returns
        -------
        str | None
            The chosen farm, guaranteed to be an item in `farms`.
        """
        while True:
            print('Choose the farm to work on: ')
            for i, farm in enumerate(farms):
                print(f'({i+1}) {farm}')
            print(f'(q) Exit application')

            user_selection = input()
            if user_selection == 'q':
                return None
            try:
                selected_index = int(user_selection)
                if selected_index in range(1, len(farms) + 1):
                    return farms[selected_index - 1]
                print('Selected index out of range.\n')
            except ValueError:
                print('Invalid selection.\n')

    def select_model(self):
        return NotImplementedError()

    def create_predictor(self):
        return NotImplementedError()

    def select_turbines(self):
        return NotImplementedError()

    def display_results(self):
        return NotImplementedError()


if __name__ == '__main__':
    app = TurbinatorApp()
    app.run()

