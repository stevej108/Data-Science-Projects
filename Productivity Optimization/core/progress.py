class ProgressReporter:

    def __init__(self):
        self.messages = []

    def update(self, icon, message):

        self.messages.append({
            "icon": icon,
            "message": message
        })

    def get_messages(self):
        return self.messages