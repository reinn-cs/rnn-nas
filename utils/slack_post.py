import requests

from config.env_config import EnvironmentConfig


class SlackPost:
    """
    A convenience class that can post updates to a Slack channel. This is convenient when running the search unattended.
    The webhook must be specified in the env.json environment configuration file in order for this to work.
    If the webhook is not specified (has a value of -1), then posting will simply be ignored.
    """

    __instance = None

    def __init__(self):
        if SlackPost.__instance != None:
            raise Exception('Trying to create new instance of existing singleton.')

        self.env = EnvironmentConfig.get_config('env_name')
        SlackPost.__instance = self

    @staticmethod
    def get_instance():
        if SlackPost.__instance is None:
            SlackPost()
            return SlackPost.__instance
        else:
            return SlackPost.__instance

    @staticmethod
    def post_success(title, value):
        message = {}
        message['title'] = title
        message['value'] = value
        SlackPost.get_instance().post_message(message, '#00ff04')

    @staticmethod
    def post_failure(title, value):
        message = {}
        message['title'] = title
        message['value'] = value
        SlackPost.get_instance().post_message(message, '#D00000')

    @staticmethod
    def post_neutral(title, value):
        message = {}
        message['title'] = title
        message['value'] = value
        SlackPost.get_instance().post_message(message, '#00c8ff')

    def post_message(self, message, color):
        if not EnvironmentConfig.get_post_slack():
            return

        url = EnvironmentConfig.get_slack_webhook()
        if url is None or url == '-1':
            return

        headers = {'Content-Type': 'application/json'}
        title = f"[{self.env}]: {message['title']}"
        value = message['value']
        payload = {"text": title, "attachments": [{"color": color, "fields": [{"value": value, "short": "false"}]}]}
        _ = requests.request("POST", url, json=payload, headers=headers)
