import { Injectable } from '@angular/core';
import { AngularFireAuth } from 'angularfire2/auth';
import * as firebase from 'firebase/app';
import {Observable} from 'rxjs';
import {err} from 'util';
import {HttpClient} from '@angular/common/http';


@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private user: Observable<firebase.User>;
  public uid: string;
  configUrl = 'http://127.0.0.1:5000/createUser/';

  fetchedUser = [];
  constructor(private firebaseAuth: AngularFireAuth, private http: HttpClient) {
    this.user = firebaseAuth.authState;
  }
  signup(email: string, password: string) {
    this.firebaseAuth.auth.createUserWithEmailAndPassword(email, password).then(value => {
      console.log('Success!', value);
      this.createInternalUser(value.user.uid);
    }).catch(err => {
      console.log('Something went wrong:', err.message);
    });
  }
  login(email: string, password: string) {
    this.firebaseAuth.auth.signInWithEmailAndPassword(email, password).then(value => {
      console.log(value.user.uid);
      this.uid = value.user.uid;
      console.log(this.fetchedUser);
    }).catch(err => {
      console.log('Something went wrong', err.message);
    });
  }
  logout() {
    this.firebaseAuth.auth.signOut();
  }
  createInternalUser(uid: string){
    this.http.get(this.configUrl + uid).subscribe(
      (response: any[]) => console.log(response),
      (error) => console.log(error)
    );
  }

}
