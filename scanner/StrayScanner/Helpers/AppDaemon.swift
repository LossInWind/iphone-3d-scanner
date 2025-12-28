//
//  AppDaemon.swift
//  StrayScanner
//
//  Created by Kenneth Blomqvist on 1/17/21.
//  Copyright © 2021 Stray Robots. All rights reserved.
//

import Foundation
import CoreData
import UIKit

class AppDaemon {
    private let dataContext: NSManagedObjectContext?

    init(appDelegate: AppDelegate) {
        dataContext = appDelegate.persistentContainer.viewContext
    }

    public func removeDeletedEntries() {
        // 在后台线程执行检查
        DispatchQueue.global(qos: .utility).async { [weak self] in
            self?.performRemoveDeletedEntries()
        }
    }
    
    private func performRemoveDeletedEntries() {
        let request = NSFetchRequest<NSManagedObject>(entityName: "Recording")
        do {
            let fetched: [NSManagedObject] = try dataContext?.fetch(request) ?? []
            let sessions = fetched.compactMap { $0 as? Recording }
            
            var hasChanges = false
            
            for session in sessions {
                if let path = session.absoluteRgbPath() {
                    let enclosingFolder = path.deletingLastPathComponent()
                    if !FileManager.default.fileExists(atPath: enclosingFolder.path) {
                        print("The dataset folder has been removed.")
                        hasChanges = true
                        DispatchQueue.main.async { [weak self] in
                            self?.removeEntry(session)
                        }
                    }
                } else {
                    print("Session \(session) does not have an rgb file.")
                }
            }
            
            if hasChanges {
                DispatchQueue.main.async {
                    NotificationCenter.default.post(name: NSNotification.Name("sessionsChanged"), object: nil)
                }
            }
        } catch let error as NSError {
            print("Something went wrong. Error: \(error), \(error.userInfo)")
        }
    }

    private func removeEntry(_ session: Recording) {
        session.deleteFiles()
        self.dataContext?.delete(session)
        do {
            try self.dataContext?.save()
        } catch let error as NSError {
            print("Could not delete recording. \(error), \(error.userInfo)")
        }
    }
}
