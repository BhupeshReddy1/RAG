from wtforms import SubmitField, FileField
from wtforms.validators import DataRequired, Email, EqualTo, Length
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, ValidationError
from wtforms import SubmitField, FileField
from wtforms.validators import DataRequired, Email, EqualTo, Length
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, ValidationError


def simple_email_check(form, field):
    if "@" not in field.data or "." not in field.data:
        raise ValidationError("Invalid email address.")


class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=50)])
    email = StringField('Email', validators=[DataRequired(), simple_email_check])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')


class QueryForm(FlaskForm):
    pdf = FileField('Upload PDF', validators=[DataRequired()])
    query = StringField('Enter Query', validators=[DataRequired()])
    submit = SubmitField('Submit')
    new_chat = SubmitField('New Chat')


class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])