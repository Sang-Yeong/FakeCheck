package com.example.fakecheck;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;

import androidx.appcompat.app.AppCompatActivity;

public class intro extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_intro);
        Handler handler = new Handler();
        handler.postDelayed(new Runnable(){
            @Override public void run() {
                Intent intent = new Intent (getApplicationContext(), MainActivity.class);
                startActivity(intent);
                finish();
            }
        },3000);
    } @Override
    protected void onPause(){
        super.onPause();
        finish();
    }
}