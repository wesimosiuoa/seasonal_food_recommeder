from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import check_password_hash, generate_password_hash
from . import app, db
from .models import User
from app.dbcon import get_con
from app.k_means import *
#from secret_key import secret_key


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend (): 
    if request.method == 'POST': 
        daytype = request.form['daytype']
        daypart = request.form['daypart']
        season = request.form['season']

        items = recommend_items(daypart,daytype,season, 10) 
        # recommend_items('Afternoon', 'Weekend', 'Spring', top_n=5)
        return render_template ('dashboard.html', items = items, daytype=daytype, daypart=daypart, season=season )
@app.route('/signin', methods=['GET', 'POST'])
def signin ():
    # return f"{secret_key}" 
    if request.method == 'POST': 
        db = get_con()
        uname = request.form['email']
        pwd = request.form['password']

        cursor = db.cursor()
        cursor.execute (" select * from users where username = %s", (uname,))
        users = cursor.fetchone()

        if users : 
            stored_password = users[3]
            if pwd == stored_password : 
                session['username'] = uname
                flash (" Login successful", 'success')
                return redirect(url_for ('dashboard'))
            else : 
                flash ('Incorrect login ', 'danger')
        else :
            flash ('User not found ')
    return render_template ('sign-in.html')

@app.route ('/dashboard')
def dashboard (): 
    if 'username' in session:
        return render_template('dashboard.html', username=session['username'])
    else : 
        flash (' Please login first', 'danger')
        return redirect(url_for('/signin'))

@app.route ('/signup', methods = ['GET', 'POST'])
def signup(): 
    if request.method == 'POST': 
        username = request.form.get('email')
        password = request.form.get('password')
        role_id= 1

        new_user = User (username = username, role_id = role_id, password = password)
        try: 
            db.session.add (new_user)
            db.session.commit()
            # flash (f' Success : User added successfully' , 'success')
            return url_for ('signin', success =' Successfully added ')
        except Exception as e:
            db.session.rollback()
            return f" There was an issue adding the user {str(e)}"
    return render_template ('sign-up.html')
@app.route('/analysis')
def analysis (): 
    os.makedirs("app/static/images", exist_ok=True)
    Seasonal_Preference_Analysis()
    return render_template ('analysis.html')
    # return render_template ('sign-up.html')
# @app.route('/signin')
# def get_sign_in ():
#     if request.method == 'POST': 
#         uname = request.form.get('email')
#         passs = request.form.get('password')

#         return f" login with {uname} and {passs}"

if __name__ == '__main__':
    app.run(debug=True)
