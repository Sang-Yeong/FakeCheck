package com.example.fakecheck;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {
    Button btnC;
    Button btnG;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btnC = (Button)findViewById(R.id.btnCam);
        btnG = (Button)findViewById(R.id.btnGal);
        btnC.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
                Intent intent = new Intent(MainActivity.this, ViewActivity.class);
                intent.putExtra("img", "cam");
                startActivity(intent);
            }
        });
        btnG.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
                Intent intent = new Intent(MainActivity.this, ViewActivity.class);
                intent.putExtra("img", "gal");
                startActivity(intent);
            }
        });
    }

}