
package com.ethicalml.kompute

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import android.widget.Toast
import android.view.View
import android.widget.EditText
import android.widget.TextView
import com.ethicalml.kompute.databinding.ActivityKomputeJniBinding
import com.ethicalml.kompute.models.Particle

class KomputeJni : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val binding = ActivityKomputeJniBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.komputeGifView.loadUrl("file:///android_asset/komputer-2.gif")

        binding.komputeGifView.getSettings().setUseWideViewPort(true)
        binding.komputeGifView.getSettings().setLoadWithOverviewMode(true)

        val successVulkanInit = initVulkan()
        if (successVulkanInit) {
            Toast.makeText(applicationContext, "Vulkan Loaded SUCCESS", Toast.LENGTH_SHORT).show()
        } else {
            binding.KomputeButton.isEnabled = false
            Toast.makeText(applicationContext, "Vulkan Load FAILED", Toast.LENGTH_SHORT).show()
        }
        Log.i("KomputeJni", "Vulkan Result: " + successVulkanInit)

        binding.predictionTextView.text = "N/A"
    }

    fun KomputeButtonOnClick(v: View) {
        val p = Particle(1.0f, 1.0f)
        val pTwo = Particle(1.5f, 1.5f)
        val particles = arrayOf(p, pTwo)
        val t = particleTest(particles, 2)
        Log.d("TESTING" , "${t[0]} ${t[1]}")
    }


    external fun initVulkan(): Boolean


    external fun particleTest(p: Array<Particle>, pTwo: Int) : FloatArray

    companion object {
        init {
            System.loadLibrary("kompute-jni")
        }
    }
}

