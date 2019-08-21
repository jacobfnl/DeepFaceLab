class BaseConfig(object):
    DEBUG = True
    TESTING = False

class productionConfig(BaseConfig):
    DEBUG = False

class DevelopmentConfig(BaseConfig):
    DEBUG = True
    Testing = True