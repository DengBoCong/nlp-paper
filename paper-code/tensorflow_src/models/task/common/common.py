from optparse import OptionParser


class CmdParser(OptionParser):
    def error(self, msg):
        print('Error!提示信息如下：')
        self.print_help()
        self.exit(0)

    def exit(self, status=0, msg=None):
        exit(status)
