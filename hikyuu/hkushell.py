#!/usr/bin/python
# -*- coding: utf8 -*-
# cp936
#
# The MIT License (MIT)
#
# Copyright (c) 2021 hikyuu.org
#
# Create on: 2021-03-09
#    Author: fasiondog

import cmd
import functools


def parse(arg):
    'Convert a series of zero or more numbers to an argument tuple'
    return tuple(map(int, arg.split()))


def shell_cmd(func):
    @functools.wraps(func)
    def wrapcmd(*args, **kargs):
        try:
            self = args[0]
            self.__class__.lineno += 1
            self.__class__.prompt = '\033[32;40mHKU [\033[0m\033[1;32;40m%s\033[0m\033[32;40m]:\033[0m ' % self.__class__.lineno
            result = func(*args, **kargs)
            print("")
            return result
        except Exception as e:
            print(e)

    return wrapcmd


class HKUShell(cmd.Cmd):
    intro = """\033[1;32;40m
    ooooo   ooooo ooooo oooo    oooo oooooo   oooo ooooo     ooo ooooo     ooo 
    `888'   `888' `888' `888   .8P'   `888.   .8'  `888'     `8' `888'     `8' 
     888     888   888   888  d8'      `888. .8'    888       8   888       8  
     888ooooo888   888   88888[         `888.8'     888       8   888       8  
     888     888   888   888`88b.        `888'      888       8   888       8  
     888     888   888   888  `88b.       888       `88.    .8'   `88.    .8'  
    o888o   o888o o888o o888o  o888o     o888o        `YbodP'       `YbodP'    
    \033[0m
    \033[33m\033[1m  Welcome to the hikyuu shell. Type help or ? to list commands.\033[0m

    """
    lineno = 1
    prompt = '\033[32;40mHKU [\033[0m\033[1;32;40m%s\033[0m\033[32;40m]:\033[0m ' % lineno
    file = None

    @shell_cmd
    def do_hello(self, arg):
        'test'
        print(arg)

    def do_quit(self, arg):
        'Stop recording, and exit:  BYE'
        print('Thank you for using hikyuu.\n')
        return True

    # ----- record and playback -----
    def do_record(self, arg):
        'Save future commands to filename:  RECORD rose.cmd'
        self.file = open(arg, 'w')

    def do_playback(self, arg):
        'Playback commands from a file:  PLAYBACK rose.cmd'
        self.close()
        with open(arg) as f:
            self.cmdqueue.extend(f.read().splitlines())

    def precmd(self, line):
        line = line.lower()
        if self.file and 'playback' not in line:
            print(line, file=self.file)
        return line

    def close(self):
        if self.file:
            self.file.close()
            self.file = None


if __name__ == "__main__":
    import colorama
    colorama.init(autoreset=True)
    HKUShell().cmdloop()