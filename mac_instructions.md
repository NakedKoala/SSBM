
Unfortunately the MacOS patch for libmelee did not work perfectly as expected, so I modified some of the files locally to make it work

in the melee package, under /melee/console.py, make the below changes to the file, and it should just work (assuming mac file systems are not drastically different)
    
    def _get_dolphin_home_path(self):
        """Return the path to dolphin's home directory"""
        if self.path:
            return self.path + "/Contents/Resources/User/"
        return ""
    
    def _get_dolphin_config_path(self):
        """ Return the path to dolphin's config directory
        (which is not necessarily the same as the home path)"""
        if self.path:
            if platform.system() == "Linux":
                # First check if the config path is here in the same directory
                if os.path.isdir(self.path + "/User/Config/"):
                    return self.path + "/User/Config/"
                # Otherwise, this must be an appimage install. Use the .config
                return str(Path.home()) + "/.config/SlippiOnline/Config/"
            else:
                return self.path + "/Contents/Resources/User/Config/"
        return ""
    
    def get_dolphin_pipes_path(self, port):
        """Get the path of the named pipe input file for the given controller port
        """
        if platform.system() == "Windows":
            return '\\\\.\\pipe\\slippibot' + str(port)
        if platform.system() == "Linux":
            # First check if the config path is here in the same directory
            if os.path.isdir(self.path + "/User/"):
                if not os.path.isdir(self.path + "/User/Pipes/"):
                    os.mkdir(self.path + "/User/Pipes/")
                return self.path + "/User/Pipes/slippibot" + str(port)
            if not os.path.isdir(str(Path.home()) + "/.config/SlippiOnline/Pipes/"):
                os.mkdir(str(Path.home()) + "/.config/SlippiOnline/Pipes/")
            return str(Path.home()) + "/.config/SlippiOnline/Pipes/slippibot" + str(port)
    
        if not os.path.isdir(self._get_dolphin_home_path() + "Pipes"):
            os.mkdir(self._get_dolphin_home_path() + "/Pipes")
    
    
        return self._get_dolphin_home_path() + "Pipes/slippibot" + str(port)