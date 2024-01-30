import os
import pwd
import shutil
import site
from subprocess import CalledProcessError, check_call

c = get_config()

c.JupyterHub.authenticator_class = "generic-oauth"

c.JupyterHub.spawner_class = "jupyterhub.spawner.LocalProcessSpawner"

# Set JupyterLab as the default interface
c.Spawner.default_url = "/lab"

c.Authenticator.admin_users = {os.environ.get("JUPYTERHUB_ADMIN_USER")}
c.JupyterHub.services = [
    {
        "name": "fastapi-backend",
        "admin": True,
        "api_token": os.environ.get("JUPYTERHUB_TOKEN"),
    }
]
c.Authenticator.allow_all = True

# Set up Keycloak authentication
c.GenericOAuthenticator.client_id = os.environ.get("JUPYTERHUB_OAUTH_CLIENT_ID")
# c.GenericOAuthenticator.client_secret = os.environ.get("OAUTH_CLIENT_SECRET")
c.GenericOAuthenticator.authorize_url = os.environ.get("JUPYTERHUB_OAUTH_AUTHORIZE_URL")
c.GenericOAuthenticator.token_url = os.environ.get("JUPYTERHUB_OAUTH_TOKEN_URL")
c.GenericOAuthenticator.userdata_url = os.environ.get("JUPYTERHUB_OAUTH_USERDATA_URL")
c.GenericOAuthenticator.oauth_callback_url = os.environ.get("JUPYTERHUB_OAUTH_CALLBACK_URL")
c.GenericOAuthenticator.login_service = os.environ.get("JUPYTERHUB_OAUTH_LOGIN_SERVICE")
# c.GenericOAuthenticator.logout_redirect_url = os.environ.get("OAUTH_LOGOUT_URL")
c.GenericOAuthenticator.username_claim = "email"
c.GenericOAuthenticator.userdata_params.state = "state"

c.GenericOAuthenticator.scope = ["openid", "email"]


def recursive_chown(path, uid, gid):
    os.chown(path, uid, gid)
    for root, dirs, files in os.walk(path):
        for directory in dirs:
            os.chown(os.path.join(root, directory), uid, gid)
        for filename in files:
            os.chown(os.path.join(root, filename), uid, gid)


def pre_spawn_hook(spawner):
    username = spawner.user.name
    try:
        check_call(["useradd", "-ms", "/bin/bash", username])
    except CalledProcessError as e:
        # Ignore the error if the user already exists (exit status 9)
        if e.returncode != 9:
            raise RuntimeError(str(e))

    sitepkgs = site.getsitepackages()

    files_dir = os.path.join(sitepkgs[0], "market_alerts")
    os.makedirs(files_dir, exist_ok=True)
    source_files = "/tmp/market_alerts"

    uid = pwd.getpwnam(username).pw_uid
    gid = pwd.getpwnam(username).pw_gid

    for file_name in os.listdir(source_files):
        source_file = os.path.join(source_files, file_name)
        dest_file = os.path.join(files_dir, file_name)
        if os.path.isdir(source_file):
            shutil.copytree(source_file, dest_file, dirs_exist_ok=True)
        else:
            shutil.copy2(source_file, dest_file)

    recursive_chown(files_dir, uid, gid)


c.Spawner.pre_spawn_hook = pre_spawn_hook


def userdata_hook(spawner, auth_state):
    spawner.environment["TWELVE_API_KEY"] = os.environ.get("TWELVE_API_KEY")
    spawner.environment["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
    spawner.environment["ALPHAVANTAGE_API_KEY"] = os.environ.get("ALPHAVANTAGE_API_KEY")


c.Spawner.auth_state_hook = userdata_hook
