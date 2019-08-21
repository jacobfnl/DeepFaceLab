class BaseConfig(object):
    """
    Base config class
    """
    DEBUG = True
    TESTING = False

class productionConfig(BaseConfig):
    """
    Production specific config
    """
    DEBUG = False

class DevelopmentConfig(BaseConfig):
    """
    Development env specific config
    """
    DEBUG = True
    Testing = True