'use client'
import React, { useState } from 'react'
import {
    Card,
    CardAction,
    CardContent,
    CardDescription,
    CardFooter,
    CardHeader,
    CardTitle,
} from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import Image from "next/image"
export default function page() {
    const data = [
        { id: 1, name: 'Chat Instance #1', description: 'This is a description for Chat Instance #1' },
        { id: 2, name: 'Chat Instance #2', description: 'This is a description for Chat Instance #2' },
        { id: 3, name: 'Chat Instance #3', description: 'This is a description for Chat Instance #3' },
    ]
    const [activeTab, setActiveTab] = useState('tiles');
    return (
        <>
            <div className="h-[70px] flex justify-between bg-white">
                <Image
                    src="/data/thumbnail_QCRI-RGB.png"
                    alt="QCRI Thumbnail"
                    width={150}
                    height={150}
                    className="object-contain ml-7 mt-2.5"
                />
            </div>

            <div className='flex flex-col items-center bg-white min-h-screen'>
                <div className='flex w-4/5 justify-start mt-2.5'>
                    <h1 className='text-4xl font-bold text-gray-800 my-8.5'>Welcome to whateverthisappisgoingtobecalled</h1>
                </div>
                <div className="gap-4 w-4/5 mt-2.5">
                    <div className="bg-green-100 rounded shadow" />
                    <div className="rounded flex justify-between px-5.5 bg-white">
                        <Button variant="outline" className="bg-gray-100 text-gray-950 hover:bg-gray-300 border-gray-300">Create New +</Button>
                        <div className="flex gap-6">
                            <Tabs defaultValue="tiles">
                                <TabsList className="bg-gray-100">
                                    <TabsTrigger value="tiles" onClick={() => setActiveTab('tiles')} className="text-gray-500 border-1 border-gray-300 data-[state=active]:bg-gray-300 data-[state=active]:text-gray-800 ">Tiles</TabsTrigger>
                                    <TabsTrigger value="cards" onClick={() => setActiveTab('cards')} className="text-gray-500 border-1 ml-1.5 border-gray-300 data-[state=active]:bg-gray-300 data-[state=active]:text-gray-800 ">Cards</TabsTrigger>
                                </TabsList>
                            </Tabs>
                        </div>
                    </div>

                    {activeTab === 'cards' && (
                        <div className="rounded grid grid-cols-3 gap-5 mt-4">
                            {data.map((item) => (
                                <Card className="bg-zinc-200 border-2 border-gray-300 shadow hover:scale-102 transition-transform duration-100" key={item.id}>
                                    <CardHeader>
                                        <CardTitle className="text-2xl text-gray-800 font-bold">{item.name}</CardTitle>
                                    </CardHeader>
                                    <CardFooter>
                                        <p className="text-gray-600">{item.description}</p>
                                    </CardFooter>
                                </Card>
                            ))}
                        </div>
                    )}
                    {activeTab === 'tiles' && (
                        <div className="rounded grid grid-cols-1 gap-5 mt-4">
                            {data.map((item) => (
                                <Card className="bg-zinc-200 border-2 border-gray-300 shadow hover:scale-102 transition-transform duration-100" key={item.id}>
                                    <CardHeader>
                                        <CardTitle className="text-2xl text-gray-800 font-bold">{item.name}</CardTitle>
                                    </CardHeader>
                                    <CardFooter>
                                        <p className="text-gray-600">{item.description}</p>
                                    </CardFooter>
                                </Card>
                            ))}
                        </div>
                    )}

                </div>
            </div>
        </>
    )
}
